import streamlit as st
import pandas as pd
import os
os.system("pip install --upgrade matplotlib")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
import plotly.graph_objects as go

from branca.colormap import linear
from shapely.geometry import shape
from streamlit_folium import folium_static
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu


def home_page(data):
    
    if data is not None:
        with st.expander("🧭 My dataset"):
            default_columns = list(data.columns)
            selected_columns = st.multiselect('Filter:', data.columns, default=default_columns)
            data_filtered = data[selected_columns].copy()
            # Format kolom 'tahun' dan 'kode_kabupaten_kota'
            for col in ['tahun', 'kode_kabupaten_kota']:
                if col in data_filtered.columns:
                    data_filtered[col] = data_filtered[col].astype(str).str.replace(',', '')  
            st.dataframe(data_filtered, use_container_width=True)
   
    
    st.markdown("<h5 style='text-align: center;'>📊 Distribusi dan Rincian Data Percluster</h5>", unsafe_allow_html=True)
    cluster_counts = data['cluster'].value_counts()
    labels = cluster_counts.index
    values = cluster_counts.values
    colors_pastel = ['#EFBC9B', '#FBF3D5', '#D6DAC8', '#9CAFAA']

    colom1, colom2 = st.columns(2)
    with colom1:
        fig = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=values, hole=0.5, opacity=0.8,
                            marker=dict(colors=colors_pastel, line=dict(color='#000000', width=2))))
        fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=12)
        
        # Mengatur ukuran plot
        fig.update_layout(width=400, height=400)

        st.plotly_chart(fig, use_container_width=True)

    with colom2:
        cols = st.columns(2) 
        for i in range(1, 5):
            cluster_df = data[data['cluster'] == i][['tahun', 'nama_kabupaten_kota']].sort_values('nama_kabupaten_kota').reset_index(drop=True)
            # Remove commas if any
            for col in cluster_df.columns:
                cluster_df[col] = cluster_df[col].astype(str).str.replace(',', '') 
            
            with cols[(i-1) // 2]:  
                st.markdown(f"**Cluster {i}**")
                st.dataframe(cluster_df, height=100, width=250)  
                
    # Membandingkan karakteristik klaster menggunakan box plot
    st.markdown("<h5 style='text-align: center;'>📊 Box Plot - Karakteristik antar cluster</h5>", unsafe_allow_html=True)
  
    col1, col2, col3 = st.columns(3)
    with col1:
        # Box plot untuk Jumlah Kasus DBD antar Klaster
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='cluster', y='jumlah_kasus', data=data, palette='Set2')  
        plt.title('Perbandingan Jumlah Kasus DBD antar cluster')
        boxplot_jumlah_kasus = plt.gcf()  
        st.pyplot(boxplot_jumlah_kasus)

    with col2:
    # Box plot untuk Kepadatan Penduduk antar Klaster
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='cluster', y='kepadatan_penduduk', data=data, palette='Set2') 
        plt.title('Kepadatan Penduduk antar cluster')
        boxplot_kepadatan_penduduk = plt.gcf()  
        st.pyplot(boxplot_kepadatan_penduduk)

    with col3:      
        # Box plot untuk Jumlah Banjir antar Klaster
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='cluster', y='jumlah_banjir', data=data, palette='Set2') 
        plt.title('Perbandingan Jumlah Banjir antar cluster')
        boxplot_jumlah_banjir = plt.gcf()  
        st.pyplot(boxplot_jumlah_banjir)

# Fungsi untuk memuat data dari Excel
def load_data():
    path = "output clustering.xlsx"
    return pd.read_excel(path)

def progress_page(data, filtered_df):
    
    # Daftar kolom yang dapat dipilih
    kolom_dapat_dipilih = ['jumlah_kasus', 'kepadatan_penduduk', 'jumlah_banjir']  

    # Pilih kolom menggunakan selectbox
    kolom = st.selectbox('Pilih Kolom', kolom_dapat_dipilih)

    # Filter berdasarkan kolom yang dipilih
    filtered_kolom = data[kolom]
    
    # Prepare the figure for plotting
    plt.figure(figsize=(10, 6))
    
    for cluster in sorted(filtered_df['cluster'].unique()):
        # Filter by cluster
        df_cluster = filtered_df[filtered_df['cluster'] == cluster]
        
        # Sort by 'nama_kabupaten_kota' or another column if necessary
        df_cluster_sorted = df_cluster.sort_values('nama_kabupaten_kota')
        
        # Extracting the names and values for plotting
        kabupaten_names = df_cluster_sorted['nama_kabupaten_kota']
        # Replace 'filtered_kolom' with the actual column name selected via selectbox
        kasus_values = df_cluster_sorted[kolom]
        
        # Plot the line for this cluster
        plt.plot(kabupaten_names, kasus_values, marker='o', label=f'Cluster {cluster}')

    # Final plot adjustments
    plt.xlabel('Nama Kabupaten/Kota')
    plt.ylabel(kolom)
    plt.title('Plot per Cluster')
    plt.legend(title='cluster')
    plt.grid(True)
    plt.xticks(rotation=90, fontsize='small')
    
    # Show the plot in Streamlit
    st.pyplot(plt)

# Buat colormap
colormap = linear.YlOrRd_09.scale(vmin=1, vmax=5).to_step(5)
colormap.index = [1, 2, 3, 4]
colormap.colors = colormap.colors[::-1]

def get_color_for_cluster(cluster):
    return colormap(cluster)

def style_function(feature, filtered_data):
    cluster = filtered_data.loc[filtered_data['nama_kabupaten_kota'] == feature['properties']['name'], 'cluster'].iloc[0]
    color = get_color_for_cluster(cluster)
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 1.5,
        'fillOpacity': 0.8
    }

def merge_data(geojson, data):
    for feature in geojson['features']:
        feature_name = feature['properties']['name']
        match = data[data['nama_kabupaten_kota'] == feature_name]
        if not match.empty:
            match = match.iloc[0]
            feature['properties']['cluster'] = int(match['cluster'])
            feature['properties']['jumlah_kasus'] = int(match['jumlah_kasus'])
            feature['properties']['kepadatan_penduduk'] = int(match['kepadatan_penduduk'])
            feature['properties']['jumlah_banjir'] = int(match['jumlah_banjir'])

def visualize_map(data, year, geojson_path, image_path):
    # Filter the data based on the selected year
    filtered_data = data[data['tahun'] == year].copy()

    # Load GeoJSON data
    with open(geojson_path, 'r') as file:
        geojson = json.load(file)

    # Merge Excel data with GeoJSON
    merge_data(geojson, filtered_data)

    # Initialize the Folium map
    m = folium.Map(location=[-6.9344694, 108.0], zoom_start=8, height=460)

    # Add GeoJSON to the map with the corresponding style function and tooltip
    folium.GeoJson(
        geojson,
        style_function=lambda feature: style_function(feature, filtered_data),
        highlight_function=lambda x: {'weight':3, 'color':'black'},
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'cluster', 'jumlah_kasus', 'kepadatan_penduduk', 'jumlah_banjir'],
            aliases=['Nama Kabupaten/Kota:', 'Cluster:', 'Jumlah Kasus:', 'Kepadatan Penduduk:', 'Jumlah Banjir:'],
            localize=True
        )
    ).add_to(m)

    # Add labels to each polygon
    for feature in geojson['features']:
        property_name = feature['properties']['name']
        matching_data = filtered_data[filtered_data['nama_kabupaten_kota'] == property_name]
        if not matching_data.empty:
            centroid = shape(feature['geometry']).centroid
            folium.Marker(
                [centroid.y, centroid.x],
                icon=folium.DivIcon(html=f'<div style="font-size: 8pt; color : black">{property_name}</div>')
            ).add_to(m)

    # Menambahkan gambar legenda di pojok kanan atas
    bounds = [[-6.7, 108.6], [-5.6, 109.85]]
  
    folium.raster_layers.ImageOverlay(
        name='Legenda',
        image=image_path,
        bounds=bounds,
        opacity=1,
        interactive=False,
        cross_origin=False,
        zindex=1
    ).add_to(m)
    
    # Render the map
    folium_static(m)


def plot_cases_over_time(df, selected_kabupaten):
    plt.figure(figsize=(14, 12))
    for kabupaten in selected_kabupaten:
        # Filter the dataframe for the selected kabupaten and sort by year
        kabupaten_data = df[df['nama_kabupaten_kota'] == kabupaten].sort_values('tahun')
        # Plot the line chart for the selected kabupaten
        plt.plot(kabupaten_data['tahun'], kabupaten_data['jumlah_kasus'], marker='o', label=kabupaten)
    plt.title('Perubahan Kasus DBD Tahun 2020-2022')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Kasus')
    plt.xticks([2020, 2021, 2022])
    plt.legend(title='Kabupaten/Kota')
    plt.grid(True)
    return plt

# Main function to run the Streamlit app
def main():
    # Load the data
    data = load_data()
    geojson_path = 'JB.json'
    image_path = '350x350.png'
    
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Analisis Cluster", "Data Visualisasi"],
            menu_icon="none",
            default_index=0,
        )

    if selected == "Analisis Cluster": 
        home_page(data)

    elif selected == "Data Visualisasi":

        # Filter Tahun
        year = st.sidebar.selectbox('Pilih tahun', sorted(data['tahun'].unique()))
        # Filter Kabupaten/Kota
        selected_kabupaten = st.sidebar.multiselect('Pilih Kabupaten/Kota', data['nama_kabupaten_kota'].unique())

        # Data filtering
        filtered_df = data[(data['tahun'] == year) & (data['nama_kabupaten_kota'].isin(selected_kabupaten))]
        
        st.write(filtered_df)

        clm1, clm2 = st.columns(2)
        with clm1:
            st.markdown(f"<h6 style='text-align: center;'>Plot Per Cluster</h6>", unsafe_allow_html=True)
            progress_page(data, filtered_df)

        with clm2:
            # Add section for line plot visualization
            st.markdown(f"<h6 style='text-align: center;'>Perubahan Kasus DBD Tahun 2020-2022</h6>", unsafe_allow_html=True)
            line_plot = plot_cases_over_time(data, selected_kabupaten)
            st.pyplot(line_plot)

        st.markdown(f"<h5 style='text-align: center;'>Peta Cluster Potensi Demam Berdarah di Jawa Barat Tahun {year}</h5>", unsafe_allow_html=True)
        # Visualize the map
        visualize_map(data, year, geojson_path, image_path)


if __name__ == '__main__':
    main()
