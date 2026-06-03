export const failureModes = [
  {
    model: 'deepseek-r1',
    total: 765,
    categories: [
      { category: 'Blindness', value: 53, percentage: '6.9', color: '#A9A9A9' },
      { category: 'Underthinking', value: 426, percentage: '55.7', color: '#4A90E2' },
      { category: 'Overthinking', value: 286, percentage: '37.4', color: '#D0021B' },
    ],
  },
  {
    model: 'gpt-5',
    total: 993,
    categories: [
      { category: 'Blindness', value: 349, percentage: '35.1', color: '#A9A9A9' },
      { category: 'Underthinking', value: 426, percentage: '42.9', color: '#4A90E2' },
      { category: 'Overthinking', value: 218, percentage: '22.0', color: '#D0021B' },
    ],
  },
  {
    model: 'Qwen/QwQ-32B',
    total: 701,
    categories: [
      { category: 'Blindness', value: 64, percentage: '9.1', color: '#A9A9A9' },
      { category: 'Underthinking', value: 372, percentage: '53.1', color: '#4A90E2' },
      { category: 'Overthinking', value: 265, percentage: '37.8', color: '#D0021B' },
    ],
  },
]
