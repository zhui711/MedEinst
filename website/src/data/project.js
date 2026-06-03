export const links = {
  paper: 'https://arxiv.org/pdf/2601.06636',
  code: 'https://github.com/zhui711/MedEinst',
  dataset: 'https://huggingface.co/datasets/zhui711/MedEinst',
  slides: `${import.meta.env.BASE_URL}slides/MedEinst_slides.pdf`,
}

export const authors = [
  {
    name: 'Wenting Chen',
    homepage: 'https://o0t1ng0o.github.io/',
  },
  {
    name: 'Guolin Huang',
    homepage: 'https://openreview.net/profile?id=~Guolin_Huang1',
  },
  {
    name: 'Wenxuan Wang',
    homepage: 'https://jarviswang94.github.io/',
  },
  {
    name: 'Zhongrui Zhu*',
    homepage: 'https://zhui711.github.io/',
  },
]

export const affiliationLine =
  'Stanford University · Shenzhen University · Renmin University of China · Xi’an Jiaotong University'

export const bibtex = `@article{medeinst,
  title = {MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis},
  author = {Chen, Wenting and Zhu, Zhongrui and Huang, Guolin and Wang, Wenxuan},
  year = {2026},
  month = jan,
  journal = {ACL},
  archiveprefix = {arXiv},
}`
