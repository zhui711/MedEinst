# MedEinst Project Page

Static Vue/Vite academic project page for:

**MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis**

The page is configured for GitHub Pages deployment under:

`https://zhui711.github.io/MedEinst/`

Accordingly, `vite.config.js` sets:

```js
base: '/MedEinst/'
```

If this site is later deployed from a repository named `MedEinst-page`, change the Vite base to `/MedEinst-page/`.

## Local Development

Install dependencies:

```sh
npm install
```

Start a local development server:

```sh
npm run dev
```

Build the static site:

```sh
npm run build
```

Preview the production build locally:

```sh
npm run preview
```

## Assets

Static figure and HTML artifacts are served from `public/`:

- `public/fig/Fig2.png`
- `public/fig/Fig3.png`
- `public/fig/fig1b.png`
- `public/html/figure1_attribution_pie.html`
- `public/html/ACL_Final_Figures_Clean.html`
- `public/slides/MedEinst_slides.pdf`

Fig. 1(b), Fig. 2, and Fig. 3 are displayed directly with responsive `object-fit: contain` styling. The Plotly result figure is embedded with an iframe to preserve the supplied interactive HTML while disabling zoom and pan controls.

## Deployment

The GitHub Actions workflow in `.github/workflows/ci.yml` builds the Vite app and deploys `dist/` with GitHub Pages. In the repository settings, set Pages source to **GitHub Actions**.
