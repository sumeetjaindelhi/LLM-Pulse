import ora from "ora";
import { detectHardware } from "../../hardware/index.js";
import { detectAllRuntimes } from "../../runtimes/index.js";
import { runDiagnostics } from "../../analysis/doctor.js";
import { theme } from "../ui/colors.js";
import { severityIcon } from "../ui/badges.js";
import { sectionHeader } from "../ui/boxes.js";

export async function doctorCommand(options: { format?: string }): Promise<void> {
  const spinner = ora({ text: "Running diagnostics...", color: "cyan" }).start();

  const [hardware, runtimes] = await Promise.all([
    detectHardware(),
    detectAllRuntimes(),
  ]);

  const report = runDiagnostics(hardware, runtimes);
  spinner.succeed("Diagnostics complete");

  if (options.format === "json") {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  console.log(sectionHeader("System Health Check"));
  console.log();

  for (const check of report.checks) {
    const icon = severityIcon(check.severity);
    console.log(`  ${icon} ${check.message}`);
    if (check.suggestion) {
      console.log(`    ${theme.muted(`→ ${check.suggestion}`)}`);
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

  console.log();
}
