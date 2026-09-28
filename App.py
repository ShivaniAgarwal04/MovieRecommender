import gradio as gr
import pandas as pd


# ============================================================
# COLLABORATIVE FILTERING TABLE
# ============================================================

def cf_table(
    user_id: int,
    n: int
) -> pd.DataFrame:

    try:

        user_id = int(user_id)
        n = int(n)

        results = recommend_cf(
            user_id,
            n
        )

        if not results:

            return pd.DataFrame(
                columns=[
                    "Rank",
                    "Title",
                    "Score"
                ]
            )

        return pd.DataFrame(
            [
                {
                    "Rank": i,
                    "Title": title,
                    "Score": round(score, 4)
                }
                for i, (title, score)
                in enumerate(results, 1)
            ],
            columns=[
                "Rank",
                "Title",
                "Score"
            ]
        )

    except Exception as e:

        return pd.DataFrame(
            [
                {
                    "Rank": "",
                    "Title": f"Error: {str(e)}",
                    "Score": ""
                }
            ]
        )


# ============================================================
# CONTENT-BASED TABLE
# ============================================================

def content_table(
    movie_id: int,
    n: int
) -> pd.DataFrame:

    try:

        movie_id = int(movie_id)
        n = int(n)

        results = recommend_content(
            movie_id,
            n
        )

        if not results:

            return pd.DataFrame(
                columns=[
                    "Rank",
                    "Title",
                    "Score"
                ]
            )

        return pd.DataFrame(
            [
                {
                    "Rank": i,
                    "Title": title,
                    "Score": round(score, 4)
                }
                for i, (title, score)
                in enumerate(results, 1)
            ],
            columns=[
                "Rank",
                "Title",
                "Score"
            ]
        )

    except Exception as e:

        return pd.DataFrame(
            [
                {
                    "Rank": "",
                    "Title": f"Error: {str(e)}",
                    "Score": ""
                }
            ]
        )


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Movie Recommendation Engine"
) as demo:

    gr.Markdown(
        "# 5FD2 Movie Recommendation Engine"
    )

    gr.Markdown(
        """
        MovieLens 100K recommendation system
        using **Collaborative Filtering** and
        **TF-IDF Content-Based Filtering**.
        """
    )

    # ========================================================
    # COLLABORATIVE FILTERING
    # ========================================================

    with gr.Tab(
        "Collaborative Filtering"
    ):

        gr.Markdown(
            """
            ### 464 User-Based Recommendations

            Enter a MovieLens user ID to get movie
            recommendations based on users with
            similar rating patterns.
            """
        )

        user_id = gr.Number(
            label="User ID",
            value=1,
            precision=0
        )

        cf_n = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of Recommendations"
        )

        cf_button = gr.Button(
            "3AF Recommend Movies"
        )

        cf_output = gr.Dataframe(
            headers=[
                "Rank",
                "Title",
                "Score"
            ],
            datatype=[
                "number",
                "str",
                "number"
            ],
            interactive=False
        )

        cf_button.click(
            fn=cf_table,
            inputs=[
                user_id,
                cf_n
            ],
            outputs=cf_output
        )

    # ========================================================
    # CONTENT BASED
    # ========================================================

    with gr.Tab(
        "Content-Based Filtering"
    ):

        gr.Markdown(
            """
            ### 3A5 Similar Movie Recommendations

            Enter a MovieLens movie ID to find movies
            with similar titles and genres.
            """
        )

        movie_id = gr.Number(
            label="Movie ID",
            value=1,
            precision=0
        )

        content_n = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of Recommendations"
        )

        content_button = gr.Button(
            "3AC Find Similar Movies"
        )

        content_output = gr.Dataframe(
            headers=[
                "Rank",
                "Title",
                "Score"
            ],
            datatype=[
                "number",
                "str",
                "number"
            ],
            interactive=False
        )

        content_button.click(
            fn=content_table,
            inputs=[
                movie_id,
                content_n
            ],
            outputs=content_output
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "CF sample:",
        cf_table(1, 5).to_dict("records")
    )

    print(
        "Content sample:",
        content_table(1, 5).to_dict("records")
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
