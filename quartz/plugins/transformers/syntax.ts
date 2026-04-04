import { QuartzTransformerPlugin } from "../types"
import rehypePrettyCode, { Options as CodeOptions, Theme as CodeTheme } from "rehype-pretty-code"
import { Root, Code } from "mdast"
import { visit } from "unist-util-visit"
import { PluggableList } from "unified"

interface Theme extends Record<string, CodeTheme> {
  light: CodeTheme
  dark: CodeTheme
}

interface Options {
  theme?: Theme
  keepBackground?: boolean
}

const defaultOptions: Options = {
  theme: {
    light: "github-light",
    dark: "github-dark",
  },
  keepBackground: false,
}

const objLinePattern =
  /^\s*(#.*|v(?:n|t)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?){1,3}|f\s+\d+(?:\/\d*)?(?:\/\d+)?(?:\s+\d+(?:\/\d*)?(?:\/\d+)?)+|[ogsl]\s+\S.*|mtllib\s+\S.*|usemtl\s+\S.*|…|\.\.\.)\s*$/

const inferFenceLanguage = (node: Code) => {
  if (node.lang) {
    return node.lang
  }

  const lines = node.value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)

  if (lines.length === 0) {
    return undefined
  }

  const looksLikeObj =
    lines.every((line) => objLinePattern.test(line)) &&
    lines.some((line) => /^(v|vn|vt|f|o|g|s|mtllib|usemtl)\b/.test(line))

  if (looksLikeObj) {
    // Python highlights numbers and separators clearly enough for Wavefront OBJ examples.
    return "python"
  }

  return undefined
}

export const SyntaxHighlighting: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts: CodeOptions = { ...defaultOptions, ...userOpts }

  return {
    name: "SyntaxHighlighting",
    markdownPlugins() {
      const plugins: PluggableList = []

      plugins.push(() => {
        return (tree: Root) => {
          visit(tree, "code", (node: Code) => {
            const inferredLanguage = inferFenceLanguage(node)
            if (inferredLanguage) {
              node.lang = inferredLanguage
            }
          })
        }
      })

      return plugins
    },
    htmlPlugins() {
      return [[rehypePrettyCode, opts]]
    },
  }
}
