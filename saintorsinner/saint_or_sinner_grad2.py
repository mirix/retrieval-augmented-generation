import rate_report2
import gradio as gr

def rating_reporting_function(entity_str, query_str, n_result):
	result = rate_report2.saint_or_sinner_function(entity_str, query_str, n_result)
	return result
    
entity = gr.Textbox(label='Enter the name of the individual or organisation you want to carry out a background check on')
keywords = gr.Textbox(label='Enter the other keywords and/or logical operators as you would in Google and then click Submit')
n_results = gr.Slider(minimum=1, maximum=30, step=1, value=10, label='Number of search results', info='Between 1 and 30 (default: 10)')

rating = gr.Image(label='Reputational Risk Rating')
report = gr.File(label='Download full report')

demo = gr.Interface(fn=rating_reporting_function, inputs=[entity, keywords, n_results], outputs=[rating, report], allow_flagging='never',
					title='😇   Saint or Sinner   😈', description='Background Checking and Reputational Risk Rating',
					theme=gr.themes.Default()
)

if __name__ == "__main__":
	demo.launch(server_name='10.56.88.201', server_port=5666) 
