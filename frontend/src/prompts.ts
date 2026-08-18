export type PromptExample = {
  lang: string
  text: string
}

/** One example per language — a spread across scripts, not the full index. */
export const PROMPT_EXAMPLES: PromptExample[] = [
  { lang: 'EN', text: 'who invented the telephone' },
  { lang: 'HI', text: 'भारत के पहले राष्ट्रपति कौन थे?' },
  { lang: 'MR', text: 'टेलिफोनचा शोध कोणी लावला?' },
  { lang: 'TA', text: 'தொலைபேசியைக் கண்டுபிடித்தவர் யார்' },
  { lang: 'BN', text: 'টেলিফোন কে আবিষ্কার করেন' },
  { lang: 'UR', text: 'ٹیلیفون کس نے ایجاد کیا' },
]
