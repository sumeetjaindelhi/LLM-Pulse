import boxen from "boxen";
import { theme } from "./colors.js";
import { VERSION, APP_NAME } from "../../core/constants.js";

export function titleBox(content: string): string {
  return boxen(content, {
    padding: 1,
    margin: { top: 1, bottom: 0, left: 0, right: 0 },
    borderStyle: "round",
    borderColor: "cyan",
    title: `${APP_NAME} v${VERSION}`,
    titleAlignment: "left",
  });
}

export function sectionHeader(title: string): string {
  return `\n  ${theme.header(title)}`;
}

export function keyValue(key: string, value: string, indent = 2): string {
  const pad = " ".repeat(indent);
  return `${pad}${theme.label(`${key} `.padEnd(6))}${theme.value(value)}`;
}

export function subLine(text: string, indent = 8): string {
  return `${" ".repeat(indent)}${theme.muted(text)}`;
}

// Box content is 72 columns at 80-column terminals; a longer line puts boxen in wrap mode and strips every indent.
export const BOX_CONTENT_COLUMNS = 72;

/** `text` broken at spaces into sub-lines that fit the box; a word longer than the line gets its own line. */
export function wrappedSubLines(text: string, indent = 8): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(" ")) {
    if (line !== "" && indent + line.length + 1 + word.length > BOX_CONTENT_COLUMNS) {
      lines.push(line);
      line = word;
    } else {
      line = line === "" ? word : `${line} ${word}`;
    }
  }
  if (line !== "") lines.push(line);
  return lines.map((l) => subLine(l, indent));
}
