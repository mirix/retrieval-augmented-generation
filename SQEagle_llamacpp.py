import SQEagle_function_lcp
import gradio as gr

def sqeagle_function(prompt):
	result, source = SQEagle_function_lcp.get_bot_response(prompt)
	return result, source
    
prompt = gr.Textbox(label='Ask away')
response = gr.Textbox(label='Answer')
sources = gr.Markdown(label='Sources')

demo = gr.Interface(fn=sqeagle_function, inputs=[prompt], outputs=[response, sources], allow_flagging='never',
					title='🐦   SQEagle   🐦', description='I find answers in en.swissquote.lu',
					theme=gr.themes.Default()
)

if __name__ == "__main__":
	demo.launch(server_name='10.56.88.201', server_port=5000) 
