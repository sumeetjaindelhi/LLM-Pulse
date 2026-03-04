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
  return `${pad}${theme.label(key.padEnd(6))}${theme.value(value)}`;
}

export function subLine(text: string, indent = 8): string {
  return `${" ".repeat(indent)}${theme.muted(text)}`;
}
