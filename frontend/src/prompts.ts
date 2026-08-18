export type PromptExample = {
  lang: string
  text: string
}

/** Curated multilingual tries — picked because they retrieve cleanly. */
export const PROMPT_EXAMPLES: PromptExample[] = [
  { lang: 'EN', text: 'what is the capital of france' },
  { lang: 'HI', text: 'भारत की राजधानी क्या है' },
  { lang: 'MR', text: 'भारताची राजधानी कोणती आहे' },
  { lang: 'GU', text: 'ફ્રાન્સની રાજધાની શું છે' },
  { lang: 'PA', text: 'ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ' },
  { lang: 'TA', text: 'பிரான்சின் தலைநகரம் எது' },
  { lang: 'UR', text: 'فرانس کا دارالحکومت کیا ہے' },
  { lang: 'BN', text: 'ফ্রান্সের রাজধানী কী' },
  { lang: 'EN', text: 'who invented the telephone' },
  { lang: 'HI', text: 'भारत के पहले राष्ट्रपति कौन थे?' },
]
