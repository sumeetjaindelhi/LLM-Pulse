import ora from "ora";
import { execa } from "execa";
import { detectHardware } from "../../hardware/index.js";
import { detectAllRuntimes } from "../../runtimes/index.js";
import { runDiagnostics, planFixes } from "../../analysis/doctor.js";
import { theme } from "../ui/colors.js";
import { severityIcon } from "../ui/badges.js";
import { sectionHeader } from "../ui/boxes.js";
import { toCsv } from "../ui/csv.js";
import { resolveOllamaHost } from "../../core/config.js";
import type { FixAction, FixPlanEntry } from "../../core/types.js";

export async function doctorCommand(options: { format?: string; host?: string; fix?: boolean; dryRun?: boolean }): Promise<void> {
  const ollamaHost = resolveOllamaHost(options.host);
  const spinner = ora({ text: "Running diagnostics...", color: "cyan" }).start();

  const [hardware, runtimes] = await Promise.all([
    detectHardware(),
    detectAllRuntimes(ollamaHost),
  ]);

  const report = runDiagnostics(hardware, runtimes);
  spinner.succeed("Diagnostics complete");

  if (options.format === "json" || options.format === "csv") {
    // Fixes only run in table mode. Say so on stderr (stdout stays parseable)
    // instead of silently dropping a flag the user explicitly passed.
    if (options.fix || options.dryRun) {
      process.stderr.write("Note: --fix/--dry-run apply to table output only — ignored with --format json/csv.\n");
    }
    if (options.format === "json") {
      console.log(JSON.stringify(report, null, 2));
    } else {
      const headers = ["label", "severity", "message", "suggestion"];
      const rows = report.checks.map((c) => [c.label, c.severity, c.message, c.suggestion ?? ""]);
      console.log(toCsv(headers, rows));
    }
    return;
  }

  console.log(sectionHeader("System Health Check"));
  console.log();

  const fixable: FixAction[] = [];
  const applyMode = options.fix && !options.dryRun;

  for (const check of report.checks) {
    const icon = severityIcon(check.severity);
    console.log(`  ${icon} ${check.message}`);
    if (check.suggestion) {
      console.log(`    ${theme.muted(`→ ${check.suggestion}`)}`);
    }
    if (check.fix) {
      fixable.push(check.fix);
      if (!applyMode) {
        console.log(`    ${theme.command(`⚡ Auto-fix available: ${check.fix.label}`)}`);
      }
    }
  }

  console.log();

  // Score display
  const scoreColor =
    report.score >= 80 ? theme.pass
    : report.score >= 60 ? theme.warning
    : theme.fail;

  console.log(`  Score: ${scoreColor(`${report.score}/100`)} — ${report.summary}`);

  if (report.topSuggestion) {
    console.log(`  ${theme.muted("Suggestion:")} ${report.topSuggestion}`);
  }

  // Quick start hint
  const ollama = runtimes.find((r) => r.name === "Ollama");
  if (ollama?.status === "running") {
    console.log(`  Run: ${theme.command("ollama pull llama3.1:8b")} to get started!`);
  } else if (!runtimes.some((r) => r.status !== "not_found")) {
    console.log(`  Get started: ${theme.command("https://ollama.com")}`);
  }

  // Dry-run mode: show exactly what --fix would execute, touch nothing.
  if (options.dryRun && fixable.length > 0) {
    console.log();
    console.log(sectionHeader("Dry Run — Planned Fixes"));
    console.log();
    renderFixPlan(planFixes(fixable));
    console.log();
    console.log(`  ${theme.muted("No changes made. Run")} ${theme.command("llm-pulse doctor --fix")} ${theme.muted("to apply.")}`);
  } else if (options.dryRun) {
    console.log();
    console.log(`  ${theme.muted("Dry run: no auto-fixable issues detected — nothing would change.")}`);
  } else if (options.fix && fixable.length > 0) {
    console.log();
    console.log(sectionHeader("Running Auto-Fixes"));
    console.log();
    await runFixes(fixable);
  } else if (!options.fix && fixable.length > 0) {
    console.log();
    console.log(`  ${theme.muted(`${fixable.length} auto-fix(es) available. Run with`)} ${theme.command("llm-pulse doctor --fix")} ${theme.muted("to apply,")} ${theme.command("--fix --dry-run")} ${theme.muted("to preview.")}`);
  }

  console.log();
}

function renderFixPlan(plan: FixPlanEntry[]): void {
  for (const entry of plan) {
    if (entry.status === "would-run") {
      console.log(`  ${theme.pass("✓")} ${entry.label} — ${entry.description}`);
      console.log(`    ${theme.command(`$ ${entry.command}`)}`);
    } else if (entry.status === "blocked") {
      console.log(`  ${theme.fail("✗")} ${entry.label} — would be blocked (binary not in fix-runner allowlist)`);
    } else {
      console.log(`  ${theme.fail("✗")} ${entry.label} — malformed fix (no argv), would be skipped`);
    }
  }
}

async function runFixes(fixes: FixAction[]): Promise<void> {
  const plan = planFixes(fixes);

  for (let i = 0; i < fixes.length; i++) {
    const fix = fixes[i];
    const spinner = ora({ text: `${fix.label}: ${fix.description}`, color: "cyan" }).start();

    try {
      // Gate on the shared plan verdict so execution can never diverge from
      // what --dry-run previewed. argv is the structured form — the display
      // `command` string may contain shell pipes that a naive split would break.
      if (plan[i].status === "malformed") {
        spinner.fail(`${fix.label}: Failed — malformed fix (no argv)`);
        continue;
      }
      if (plan[i].status === "blocked") {
        spinner.fail(
          `${fix.label}: Blocked — "${fix.argv[0]}" is not in the fix-runner allowlist`,
        );
        continue;
      }
      const cmd = fix.argv[0];
      const args = fix.argv.slice(1);

      // For background services like `ollama serve`, spawn detached and verify.
      if (cmd === "ollama" && args[0] === "serve") {
        const child = execa(cmd, args, {
          detached: true,
          stdio: "ignore",
        });
        child.unref();
        // Wait a moment for the server to start
        await new Promise((r) => setTimeout(r, 2000));

        // Verify it started
        try {
          const res = await fetch(`${resolveOllamaHost()}/api/version`, {
            signal: AbortSignal.timeout(3000),
          });
          if (res.ok) {
            spinner.succeed(`${fix.label}: Ollama is now running`);
          } else {
            spinner.warn(`${fix.label}: Started but API not responding yet — may need a moment`);
          }
        } catch {
          spinner.warn(`${fix.label}: Started process — may take a moment to become ready`);
        }
        continue;
      }

      // For install/pull commands, run and stream output
      await execa(cmd, args, { timeout: 300000, stdio: "pipe" });
      spinner.succeed(`${fix.label}: Done`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      spinner.fail(`${fix.label}: Failed — ${msg}`);
    }
  }
}
