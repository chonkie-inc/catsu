# Catsu Models Website

A clean, minimal database of embedding models built with Astro, React, and TanStack Table.

## 🚀 Development

```bash
npm install
npm run dev
```

Open [http://localhost:4321](http://localhost:4321) in your browser.

## 📦 Building for Production

```bash
npm run build
```

This creates a `dist/` directory with your built site optimized for Cloudflare Pages.

## 🌐 Deploying to Cloudflare Pages

### Option 1: Deploy via GitHub (Recommended)

1. Push this code to GitHub
2. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
3. Navigate to **Workers & Pages** → **Create Application** → **Pages**
4. Connect your GitHub repository
5. Configure build settings:
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
   - **Root directory**: `website`
6. Click **Save and Deploy**

Cloudflare will auto-deploy on every push to main!

### Option 2: Deploy via Wrangler CLI

```bash
# Install Wrangler
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Build and deploy
npm run build
npx wrangler pages deploy dist --project-name=catsu-models
```

## 🔄 Updating Model Data

The models are loaded from `src/models.json`, which is automatically copied from `../src/catsu/data/models.json` during setup.

To update the models:
1. Update `/src/catsu/data/models.json` in the main catsu project
2. Copy it to `website/src/models.json`
3. Rebuild: `npm run build`

Or set up a build script to automate this.

## 🎨 Tech Stack

- **Astro** - Static site generator
- **React** - For interactive components
- **TanStack Table** - Powerful table with sorting/filtering
- **Tailwind CSS** - Styling
- **Cloudflare Pages** - Hosting

## 📁 Project Structure

```
website/
├── src/
│   ├── components/
│   │   └── ModelsTable.tsx    # Main table component
│   ├── pages/
│   │   └── index.astro         # Home page
│   ├── styles/
│   │   └── global.css          # Global styles
│   └── models.json             # Model data
├── public/                     # Static assets
├── astro.config.mjs            # Astro configuration
└── package.json
```

## 🔗 Custom Domain

To use a custom domain like `models.catsu.dev`:

1. Go to your Cloudflare Pages project
2. Navigate to **Custom domains**
3. Click **Set up a custom domain**
4. Enter your domain (e.g., `models.catsu.dev`)
5. Cloudflare will automatically configure DNS

Done! 🎉
