import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import threading
from PIL import Image, ImageTk, ImageDraw, ImageFont
import requests
from io import BytesIO
import time

import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

# --- CONFIGURATION & STYLING ---
MODELS_DIR = "stock_models/"
os.makedirs(MODELS_DIR, exist_ok=True)
PROFILE_CACHE = {}

# --- App Colors ---
COLORS = {
    "bg_dark": "#242424",
    "bg_light": "#2B2B2B",
    "border": "#404040",
    "text": "#DCE4EE",
    "text_secondary": "#AEAEAE",
    "green": "#2E8B57",
    "red": "#B22222",
    "blue": "#3A7EBF",
    "cyan": "#00FFFF"
}

# --- BACKEND LOGIC ---

def get_sp500_tickers(progress_callback=None):
    """
    Fetches the list of S&P 500 tickers, gets their market capitalization,
    and returns the top 50 tickers by market cap.
    Includes a User-Agent header to avoid HTTP 403 Forbidden error.
    """
    if progress_callback: progress_callback("Fetching S&P 500 ticker list...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        sp500_df = tables[0]
        sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-', regex=False)
        all_tickers = sp500_df['Symbol'].tolist()
    except Exception as e:
        if progress_callback: progress_callback(f"ERROR: Could not fetch tickers: {e}")
        return []

    if progress_callback: progress_callback(f"Found {len(all_tickers)} tickers. Fetching market caps...")
    
    ticker_market_caps = []
    for i, symbol in enumerate(all_tickers):
        try:
            ticker_info = yf.Ticker(symbol).info
            market_cap = ticker_info.get('marketCap', 0)
            if market_cap: ticker_market_caps.append((symbol, market_cap))
            if progress_callback and (i + 1) % 25 == 0:
                progress_callback(f"    -> Scanned {i + 1}/{len(all_tickers)} companies...")
            time.sleep(0.02)
        except Exception:
            if progress_callback: progress_callback(f"Warning: Could not fetch info for {symbol}.")

    sorted_tickers = sorted(ticker_market_caps, key=lambda item: item[1], reverse=True)
    if progress_callback: progress_callback("Finished fetching market caps. Selected top 50.")
    return [ticker[0] for ticker in sorted_tickers[:50]]

def get_and_cache_profiles(tickers, progress_callback):
    progress_callback("Caching company profiles...")
    for symbol in tickers:
        if symbol not in PROFILE_CACHE:
            try:
                ticker_info = yf.Ticker(symbol).info
                PROFILE_CACHE[symbol] = {
                    'name': ticker_info.get('longName', symbol),
                    'logo': ticker_info.get('logo_url', '')
                }
            except Exception:
                PROFILE_CACHE[symbol] = {'name': symbol, 'logo': ''}
    progress_callback("Finished caching profiles.")

def create_features_for_df(df_in):
    """Generates features for a given DataFrame."""
    df = df_in.copy()
    df['c'] = df['Close']
    for i in [5, 10, 20, 30]:
        df[f'close_lag_{i}'] = df['c'].shift(i)
        df[f'ma_{i}'] = df['c'].rolling(window=i).mean()
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, 1e-6))
    df['rsi_14'] = 100 - (100 / (1 + rs))
    return df

def train_model_for_symbol(symbol, progress_callback):
    try:
        df_historical = yf.download(symbol, period="5y", interval="1d", progress=False, auto_adjust=False)
        if df_historical.empty:
            progress_callback(f"No data for {symbol}, skipping.")
            return
        
        df_with_features = create_features_for_df(df_historical)
        df_with_features['future_price'] = df_with_features['c'].shift(-1)
        df_with_features = df_with_features.dropna()
        
        feature_names = ['close_lag_5', 'close_lag_10', 'close_lag_20', 'close_lag_30', 'ma_5', 'ma_10', 'ma_20', 'ma_30', 'rsi_14']
        X, y = df_with_features[feature_names], df_with_features['future_price']

        if X.empty:
            progress_callback(f"Not enough data for {symbol}, skipping.")
            return
            
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        joblib.dump({'model': model, 'feature_names': feature_names}, f"{MODELS_DIR}{symbol}_model.joblib")
        progress_callback(f"Successfully trained {symbol}")
    except Exception as e:
        progress_callback(f"ERROR training {symbol}: {e}")

def predict_for_symbol(symbol, data):
    try:
        model_file = f"{MODELS_DIR}{symbol}_model.joblib"
        if not os.path.exists(model_file):
            return {"symbol": symbol, "error": "Model not trained"}

        saved_object = joblib.load(model_file)
        model, feature_names = saved_object['model'], saved_object['feature_names']

        df_with_features = create_features_for_df(data)
        latest_features = df_with_features[feature_names].iloc[[-1]]

        if latest_features.isnull().values.any():
            return {"symbol": symbol, "error": "Could not generate features"}
            
        predicted_price = model.predict(latest_features)[0]
        current_price = data['Close'].iloc[-1]
        profile = PROFILE_CACHE.get(symbol, {'name': symbol, 'logo': ''})
        
        return {
            "symbol": symbol, "name": profile['name'], "logo": profile['logo'],
            "current_price": current_price, "predicted_price": predicted_price
        }
    except Exception as e:
        return {"symbol": symbol, "error": f"Prediction error: {e}"}

# --- GUI APPLICATION CLASS ---

class StockPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Stock Predictor")
        self.geometry("1920x1080")
        ctk.set_appearance_mode("dark")
        self.configure(fg_color=COLORS["bg_dark"])

        self.tickers, self.logo_cache = [], {}

        # --- Main Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=COLORS["bg_light"])
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.title_label = ctk.CTkLabel(self.sidebar_frame, text="AI Dashboard", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.train_button = ctk.CTkButton(self.sidebar_frame, text="Train Models", command=self.start_training, state="disabled", height=40)
        self.train_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.predict_button = ctk.CTkButton(self.sidebar_frame, text="Predict All Stocks", command=self.start_prediction, height=40,
                                            fg_color=COLORS["green"], hover_color=COLORS["red"], state="disabled")
        self.predict_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame, progress_color=COLORS["blue"])
        self.progress_bar.set(0)
        self.log_textbox = ctk.CTkTextbox(self.sidebar_frame, height=150, fg_color=COLORS["bg_dark"], border_color=COLORS["border"], border_width=1)
        self.log_textbox.grid(row=5, column=0, padx=20, pady=20, sticky="sew")

        # --- Main Content Area ---
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_view.grid_rowconfigure(0, weight=1)
        self.main_view.grid_columnconfigure(0, weight=1)

        self.dashboard_frame = ctk.CTkScrollableFrame(self.main_view, corner_radius=0, fg_color="transparent")
        self.graph_view_frame = ctk.CTkFrame(self.main_view, fg_color="transparent")
        
        self.show_dashboard_view()
        self.load_initial_data()

    # --- View Management ---
    def show_dashboard_view(self):
        self.graph_view_frame.grid_forget()
        self.dashboard_frame.grid(row=0, column=0, sticky="nsew")
        if not self.dashboard_frame.winfo_children():
            self.populate_dashboard_with_placeholder()

    def populate_dashboard_with_placeholder(self):
        self.clear_dashboard()
        placeholder = ctk.CTkLabel(self.dashboard_frame, text='Click "Predict All Stocks" to see the dashboard',
                                   font=ctk.CTkFont(size=18), text_color=COLORS["text_secondary"])
        placeholder.pack(pady=100)

    def show_graph_view(self, symbol):
        self.dashboard_frame.grid_forget()
        self.graph_view_frame.grid(row=0, column=0, sticky="nsew")
        for widget in self.graph_view_frame.winfo_children(): widget.destroy()

        # Layout for graph view
        self.graph_view_frame.grid_rowconfigure(2, weight=1)
        self.graph_view_frame.grid_columnconfigure(0, weight=1)
        
        # Top bar with Back button and Title
        top_frame = ctk.CTkFrame(self.graph_view_frame, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        back_button = ctk.CTkButton(top_frame, text="< Back to Dashboard", command=self.show_dashboard_view, fg_color=COLORS["border"])
        back_button.pack(side="left")
        title = ctk.CTkLabel(top_frame, text=f"Historical Analysis for {symbol}", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(side="left", expand=True)

        # Info panel for prices
        info_frame = ctk.CTkFrame(self.graph_view_frame, fg_color="transparent")
        info_frame.grid(row=1, column=0, sticky="ew", pady=5)
        self.current_price_label = ctk.CTkLabel(info_frame, text="Current: --", font=ctk.CTkFont(size=16))
        self.predicted_price_label = ctk.CTkLabel(info_frame, text="Predicted: --", font=ctk.CTkFont(size=16, weight="bold"))
        self.change_label = ctk.CTkLabel(info_frame, text="Change: --", font=ctk.CTkFont(size=16, weight="bold"))
        self.current_price_label.pack(side="left", padx=20)
        self.predicted_price_label.pack(side="left", padx=20)
        self.change_label.pack(side="left", padx=20)

        # Container for the chart
        graph_container = ctk.CTkFrame(self.graph_view_frame, fg_color=COLORS["bg_light"])
        graph_container.grid(row=2, column=0, sticky="nsew")
        
        threading.Thread(target=self.fetch_and_display_graph, args=(symbol, graph_container), daemon=True).start()

    # --- Data & Threading Logic ---
    def load_initial_data(self):
        # FIX: Define a full function for the thread target for clarity and to avoid the syntax error.
        def task():
            self.log_progress("Initializing...")
            # FIX: Use standard assignment '=' instead of the walrus operator ':='
            self.tickers = get_sp500_tickers(self.log_progress)
            self.log_progress(f"Identified top {len(self.tickers)} stocks.")
            get_and_cache_profiles(self.tickers, self.log_progress)
            self.log_progress("Initialization Complete. Ready to train or predict.")
            # Safely schedule the UI update on the main thread
            self.after(0, lambda: (
                self.train_button.configure(state="normal"),
                self.predict_button.configure(state="normal")
            ))
        
        threading.Thread(target=task, daemon=True).start()

    def start_training(self):
        self.train_button.configure(state="disabled")
        self.predict_button.configure(state="disabled")
        self.progress_bar.grid(row=3, column=0, padx=20, pady=(10,0), sticky="ew")
        self.log_textbox.delete("1.0", "end")
        self.log_progress("Starting training...")
        
        def task():
            for i, symbol in enumerate(self.tickers):
                train_model_for_symbol(symbol, self.log_progress)
                self.after(0, lambda p=(i + 1) / len(self.tickers): self.progress_bar.set(p))
            self.after(0, self.training_complete)
        threading.Thread(target=task, daemon=True).start()

    def training_complete(self):
        self.log_progress("--- ALL TRAINING COMPLETE ---")
        self.train_button.configure(state="normal")
        self.predict_button.configure(state="normal")
        self.progress_bar.grid_forget()

    def start_prediction(self):
        self.predict_button.configure(state="disabled", text="Predicting...")
        self.clear_dashboard()
        loading_label = ctk.CTkLabel(self.dashboard_frame, text="Loading Predictions...", font=ctk.CTkFont(size=18))
        loading_label.pack(pady=100)

        def task():
            try:
                # OPTIMIZATION: Download all data in one batch
                all_data = yf.download(self.tickers, period="1y", interval="1d", progress=False, auto_adjust=False)
                if all_data.empty:
                    self.after(0, lambda: messagebox.showerror("Error", "Could not download any market data."))
                    self.after(0, self.prediction_complete)
                    return
                
                # Predict for each stock using its slice of the downloaded data
                predictions = [predict_for_symbol(s, all_data.loc[:, (slice(None), s)].droplevel(1, axis=1)) for s in self.tickers]
                
                self.after(0, lambda: self.populate_dashboard(predictions))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Prediction Error", f"An error occurred: {e}"))
            finally:
                self.after(0, self.prediction_complete)

        threading.Thread(target=task, daemon=True).start()
    
    def prediction_complete(self):
        self.predict_button.configure(state="normal", text="Predict All Stocks")

    # --- UI Drawing & Helpers ---
    def log_progress(self, message):
        self.after(0, lambda: (self.log_textbox.insert("end", message + "\n"), self.log_textbox.see("end")))

    def clear_dashboard(self):
        for widget in self.dashboard_frame.winfo_children():
            widget.destroy()

    def populate_dashboard(self, predictions):
        self.clear_dashboard()
        
        # Sort predictions by symbol to maintain a consistent order
        predictions.sort(key=lambda p: p.get('symbol', ''))

        # Create a grid for the cards
        for i, data in enumerate(predictions):
            row, col = i // 4, i % 4
            self.dashboard_frame.grid_columnconfigure(col, weight=1)
            self.create_stock_card(self.dashboard_frame, data, row, col)

    def create_stock_card(self, parent, data, row, col):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color=COLORS["bg_light"], border_color=COLORS["border"], border_width=1)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        symbol = data.get('symbol', 'N/A')
        
        # --- Card Layout (Grid) ---
        card.grid_columnconfigure(1, weight=1)

        # --- Row 0: Logo and Symbol/Name ---
        logo_img = self.get_image_from_url(symbol, data.get('logo'))
        logo_label = ctk.CTkLabel(card, text="", image=logo_img)
        logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        name_frame = ctk.CTkFrame(card, fg_color="transparent")
        name_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")
        
        symbol_label = ctk.CTkLabel(name_frame, text=symbol, font=ctk.CTkFont(size=18, weight="bold"))
        symbol_label.pack(anchor="w")
        
        company_name = (data.get('name', 'N/A') or 'N/A')
        company_label = ctk.CTkLabel(name_frame, text=company_name[:25] + ('...' if len(company_name)>25 else ''), 
                                     text_color=COLORS["text_secondary"], anchor="w")
        company_label.pack(anchor="w", fill="x")

        # --- Row 1: Content (Prices or Error) ---
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        if "error" in data:
            error_label = ctk.CTkLabel(content_frame, text=data["error"], text_color="orange", wraplength=200)
            error_label.pack(expand=True, pady=20)
        else:
            current_p = data.get('current_price', 0)
            predicted_p = data.get('predicted_price', 0)
            change = predicted_p - current_p
            change_pct = (change / current_p) * 100 if current_p != 0 else 0
            color = COLORS["green"] if change >= 0 else COLORS["red"]
            
            # --- Prices (Grid within content_frame) ---
            content_frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(content_frame, text="Current:", anchor="w", text_color=COLORS["text_secondary"]).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(content_frame, text=f"${current_p:.2f}", anchor="e").grid(row=0, column=1, sticky="e")
            
            ctk.CTkLabel(content_frame, text="Predicted:", anchor="w", text_color=color, font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, sticky="w")
            ctk.CTkLabel(content_frame, text=f"${predicted_p:.2f}", text_color=color, font=ctk.CTkFont(weight="bold"), anchor="e").grid(row=1, column=1, sticky="e")

            # --- Row 2: Change % ---
            change_label = ctk.CTkLabel(card, text=f"{change:+.2f} ({change_pct:+.2f}%)", text_color=color, font=ctk.CTkFont(size=18, weight="bold"))
            change_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="e")

        # --- Interactivity ---
        def on_enter(e): card.configure(fg_color=COLORS["border"])
        def on_leave(e): card.configure(fg_color=COLORS["bg_light"])
        
        if "error" not in data:
            # Make all widgets in the card clickable
            all_widgets = [card, logo_label, name_frame, symbol_label, company_label, content_frame] + content_frame.winfo_children()
            if 'change_label' in locals(): all_widgets.append(change_label)
            
            for widget in all_widgets:
                widget.bind("<Button-1>", lambda event, s=symbol: self.show_graph_view(s))
                widget.bind("<Enter>", on_enter)
                widget.bind("<Leave>", on_leave)

    def fetch_and_display_graph(self, symbol, container):
        try:
            df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=False)
            if df.empty:
                self.after(0, lambda: messagebox.showerror("Error", "No valid data to plot."))
                return

            prediction_data = predict_for_symbol(symbol, df)
            if "error" in prediction_data:
                self.after(0, lambda err=prediction_data['error']: messagebox.showerror("Error", err))
                return

            predicted_price = prediction_data['predicted_price']
            current_price = prediction_data['current_price']
            
            mc = mpf.make_marketcolors(up=COLORS["green"], down=COLORS["red"], inherit=True)
            s  = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds', gridstyle='--')
            
            pred_line = [predicted_price] * len(df.index)
            add_plots = [mpf.make_addplot(pred_line, color=COLORS["cyan"], linestyle='--', width=1.5)]
            
            fig, axlist = mpf.plot(df, type='candle', style=s, ylabel='Price (USD)', mav=(50, 200),
                                   volume=True, returnfig=True, addplot=add_plots, figsize=(12, 7))
            
            # --- ADD LEGEND ---
            ax = axlist[0]
            handles, labels = ax.get_legend_handles_labels()
            # Create a proxy artist for the prediction line
            pred_proxy = Line2D([0], [0], linestyle='--', color=COLORS["cyan"], linewidth=1.5, label='AI Prediction')
            handles.append(pred_proxy)
            ax.legend(handles=handles)
            
            def update_ui():
                change = predicted_price - current_price
                change_pct = (change / current_price) * 100 if current_price != 0 else 0
                color = COLORS["green"] if change >= 0 else COLORS["red"]
                self.current_price_label.configure(text=f"Current: ${current_price:.2f}")
                self.predicted_price_label.configure(text=f"Predicted: ${predicted_price:.2f}", text_color=color)
                self.change_label.configure(text=f"Change: {change:+.2f} ({change_pct:+.2f}%)", text_color=color)
                
                for widget in container.winfo_children(): widget.destroy()
                canvas = FigureCanvasTkAgg(fig, master=container)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.after(0, update_ui)
        except Exception as e:
            self.after(0, lambda err=e: messagebox.showerror("Graph Error", f"An unexpected error occurred: {err}"))
            
    def get_image_from_url(self, symbol, url, size=(40, 40)):
        cache_key = f"{url}_{size}"
        if url and cache_key in self.logo_cache: return self.logo_cache[cache_key]
        
        if url and url.startswith("http"):
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                img_data = BytesIO(response.content)
                ctk_img = ctk.CTkImage(light_image=Image.open(img_data), size=size)
                self.logo_cache[cache_key] = ctk_img
                return ctk_img
            except Exception:
                pass # Fall through to placeholder on any error
        
        return self.create_placeholder_logo(symbol, size)
    
    def create_placeholder_logo(self, symbol, size=(40, 40)):
        cache_key = f"placeholder_{symbol}_{size}"
        if cache_key in self.logo_cache: return self.logo_cache[cache_key]

        letter = symbol[0] if symbol else '?'
        img = Image.new('RGB', size, color=COLORS["bg_dark"])
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arialbd.ttf", size[0] // 2)
        except IOError:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), letter, font=font)
        textwidth, textheight = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (size[0] - textwidth) / 2, (size[1] - textheight) / 2
        draw.text((x, y), letter, fill=COLORS["text"], font=font)
        
        ctk_img = ctk.CTkImage(light_image=img, size=size)
        self.logo_cache[cache_key] = ctk_img
        return ctk_img

if __name__ == "__main__":
    app = StockPredictorApp()
    app.mainloop()