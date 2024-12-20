import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from spark_setup import create_spark_session, load_data, file_paths

############################
# Caching and Setup
############################

@st.cache_data(show_spinner=False)
def get_model(model_path="embedding_model"):
    if not os.path.exists(model_path):
        model = SentenceTransformer('all-mpnet-base-v2')
        model.save(model_path)
    return SentenceTransformer(model_path)

@st.cache_data(show_spinner=False)
def get_embeddings(embedding_path="embeddings.npy"):
    return np.load(embedding_path)

@st.cache_resource(show_spinner=False)
def get_spark_session():
    return create_spark_session()

@st.cache_resource(show_spinner=False)
def get_data(_spark):
    return load_data(_spark, file_paths)

@st.cache_data(show_spinner=False)
def get_index_to_pub_id(_publications_spark_df):
    pub_ids = _publications_spark_df.select("publication_id").rdd.map(lambda x: x.publication_id).collect()
    return {idx: pub_id for idx, pub_id in enumerate(pub_ids)}

############################
# Main Code
############################

embeddings = get_embeddings()
model = get_model()

spark = get_spark_session()
dataframes = get_data(spark)

data = dataframes["geospatial_clustering_data"]
publications_df = dataframes["clustering"]

# Create the mapping from embedding index to publication_id
index_to_pub_id = get_index_to_pub_id(publications_df)

# Rename clusters as fields of study
field_topics = {
    0: "Phylogenetics and Species Diversity",
    1: "Advanced Materials and Nanotechnology",
    2: "Bioactive Compounds and Antioxidant Studies",
    3: "Catalysis and Energy Conversion",
    4: "Machine Learning and Image Processing",
    5: "Clinical and Epidemiological Studies",
    6: "Social and Behavioral Research",
    7: "Environmental Risk and Water Management",
    8: "Microbiology and Antibiotic Resistance",
    9: "Systems Engineering and Optimization",
    10: "Virology and Infectious Diseases",
    11: "Oral and Dental Research",
    12: "Surgery and Clinical Outcomes",
    13: "Composite Materials and Structural Engineering",
    14: "Cancer Research and Cellular Mechanisms",
    15: "Particle Physics and Cosmology",
    16: "Psychiatry and Cognitive Disorders"
}

# Page configuration
st.set_page_config(
    page_title="🌏 Chulalongkorn University Global Collaboration Explorer",
    layout="wide",
    page_icon="🌏"
)

# Initialize variables
field_id, field_name = -1, "All Fields"
keyword = None

# Sidebar
with st.sidebar:
    st.title("🌟 Global Collaboration Explorer")
    st.markdown("""
        **Explore Chulalongkorn University's global academic collaborations**  
        Use the options below to choose a field of study or explore by keyword.
    """)

    # Add a search mode radio button
    search_mode = st.radio(
        "Exploration Mode:",
        options=["Explore by Field of Study", "Explore by Keyword"],
        index=0
    )

    if search_mode == "Explore by Field of Study":
        st.markdown("#### 🎓 Select a Field of Study")
        if "selected_field" not in st.session_state:
            st.session_state.selected_field = -1

        search_query = st.selectbox(
            "Field of Study:",
            options=[(-1, "All Fields")] + list(field_topics.items()),
            format_func=lambda x: "All Fields" if x[0] == -1 else f"Field {x[0]}: {x[1]}",
            index=st.session_state.selected_field + 1
        )
        field_id, field_name = search_query
        st.session_state.selected_field = field_id

        # Filter data based on selected field
        if field_id == -1:
            filtered_map_data_spark = data
        else:
            filtered_map_data_spark = data.filter(F.col("cluster") == field_id)

    elif search_mode == "Explore by Keyword":
        st.markdown("#### 🔍 Enter a Keyword")
        keyword = st.text_input("Keyword:")
        if keyword:
            input_embedding = model.encode(keyword)
            cos_similarities = cosine_similarity([input_embedding], embeddings)[0]

            # Create similarity DataFrame
            similarity_df = pd.DataFrame({
                "publication_id": [index_to_pub_id[i] for i in range(len(embeddings))],
                "similarity": cos_similarities
            })

            # Threshold filtering
            similarity_threshold = 0.38
            similarity_df = similarity_df[similarity_df["similarity"] >= similarity_threshold]

            if similarity_df.empty:
                filtered_map_data_spark = data.limit(0)
            else:
                # Convert to Spark DF and join all matched publications
                similarity_spark_df = spark.createDataFrame(similarity_df)
                joined_df = data.join(similarity_spark_df, on="publication_id", how="inner")

                # Sort by similarity descending
                filtered_map_data_spark = joined_df.orderBy(F.col("similarity").desc())
        else:
            filtered_map_data_spark = data.limit(0)

    # Function to get unique affiliation count as points
    def get_country_points(_filtered_spark_df):
        return (
            _filtered_spark_df.groupBy("country")
            .agg(F.countDistinct("affiliation_id").alias("points"))
            .orderBy(F.col("points").desc())
        )

    country_points_spark = get_country_points(filtered_map_data_spark)
    filtered_map_data_pd = country_points_spark.toPandas()

    def get_dynamic_country_options(pdf):
        return [("All Countries", 0)] + [(row["country"], row["points"]) for _, row in pdf.iterrows()]

    if "selected_country" not in st.session_state:
        st.session_state.selected_country = "All Countries"

    country_options = get_dynamic_country_options(filtered_map_data_pd)

    selected_country = st.selectbox(
        "Select a Country:",
        options=country_options,
        format_func=lambda x: f"{x[0]} ({x[1]} unique affiliations)" if x[0] != "All Countries" else "All Countries",
        index=0
    )

    selected_country_name = selected_country[0]
    st.session_state.selected_country = selected_country_name

    # Statistics Table Section
    st.markdown("#### 📊 Show Country Statistics")
    show_stats = st.checkbox("Show Table", value=True)

# Main Title and Description
st.title("🌏 Chulalongkorn University's Global Research Collaborations")

if search_mode == "Explore by Field of Study":
    st.markdown(
        f"**Exploring collaborations in:** {'All Fields' if field_id == -1 else field_name} "
        f"**|** {'All Countries' if selected_country_name == 'All Countries' else selected_country_name}"
    )
else:
    st.markdown(
        f"**Exploring collaborations by keyword:** {'None' if not keyword else keyword} "
        f"**|** {'All Countries' if selected_country_name == 'All Countries' else selected_country_name}"
    )

# Filter by selected country if needed
if selected_country_name != "All Countries":
    filtered_map_data_spark = filtered_map_data_spark.filter(F.col("country") == selected_country_name)
    filtered_map_data_pd = (
        filtered_map_data_spark.groupBy("country")
        .agg(F.countDistinct("affiliation_id").alias("points"))
        .orderBy(F.col("points").desc())
        .toPandas()
    )

if search_mode == "Explore by Field of Study":
    title_text = f"Chulalongkorn University's Global Collaborations by {'All Fields' if field_id == -1 else field_name}"
else:
    title_text = "Chulalongkorn University's Global Collaborations by Keyword"

fig = px.choropleth(
    filtered_map_data_pd,
    locations="country",
    locationmode="country names",
    color="points",
    color_continuous_scale="Greens",
    title=title_text,
    labels={'points': 'Unique Affiliations'},
)

fig.update_geos(
    showcountries=True,
    countrycolor="Black",
    showcoastlines=True,
    coastlinecolor="Gray",
    showland=True,
    landcolor="white",
    showocean=True,
    oceancolor="lightblue",
    projection_type="natural earth"
)

fig.update_layout(
    title_font=dict(size=24, family="Arial"),
    margin={"r": 10, "t": 50, "l": 10, "b": 10},
    coloraxis_colorbar=dict(
        title="Unique Affiliations",
        title_font=dict(size=16, family="Arial"),
        tickfont=dict(size=12, family="Arial"),
    )
)

# Display the map
st.plotly_chart(fig, use_container_width=True)

# Show top 10 rows
top_10_pd = filtered_map_data_spark.limit(10).toPandas()

# Select only needed columns for preview and rename them
# Original columns: header -> affiliation, city -> city, country -> country, title_x -> title
display_df = top_10_pd[["header", "city", "country", "title_x"]].copy()
display_df.rename(columns={
    "header": "affiliation",
    "city": "city",
    "country": "country",
    "title_x": "title"
}, inplace=True)

st.markdown("### 📜 Example Papers from Chulalongkorn University and Its Partners")
st.dataframe(display_df)

# Display Country Statistics if enabled
if show_stats:
    st.markdown("---")
    st.subheader("🌐 Country Statistics (Unique Affiliations)")
    if filtered_map_data_pd.empty:
        st.write("No data available for the selected filters.")
    else:
        st.dataframe(filtered_map_data_pd.style.format(precision=0).set_properties(**{'text-align': 'left'}))
