import chalk from "chalk";

export const theme = {
  // Headers and emphasis
  header: chalk.cyan.bold,
  subheader: chalk.white.bold,
  label: chalk.gray,
  value: chalk.white,
  muted: chalk.dim,

  // Status
  pass: chalk.green,
  warning: chalk.yellow,
  fail: chalk.red,
  info: chalk.blue,

  // Special
  brand: chalk.cyan,
  highlight: chalk.magenta,
  number: chalk.yellow,
  command: chalk.green,
};
