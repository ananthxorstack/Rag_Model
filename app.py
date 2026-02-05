import flet as ft
import random
import time
import os
import shutil
from src.services.llm_service import LLMService
from src.services.vector_store import VectorStore
from src.services.document_processor import DocumentProcessor
from src.config.settings import settings
from src.config.constants import GENERIC_ERROR_MESSAGE, SOFT_REFUSAL_MESSAGES



# Initialize Services (Moved logic to main for session isolation)
# Initialize Services (Moved logic to main for session isolation)
class ServiceManager:
    def __init__(self, session_id=None):
        self.llm_service = LLMService() # This could be shared, but fine to re-init
        self.vector_store = VectorStore(self.llm_service, session_id=session_id)
        self.doc_processor = DocumentProcessor()

# Removed global instance to enforce session isolation
# services = ServiceManager()

def main(page: ft.Page):
    page.title = "Local RAG Assistant"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.window_width = 1000
    page.window_height = 800

    # State
    chat_history = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    new_message = ft.TextField(
        hint_text="Ask a question about your documents...",
        autofocus=True,
        shift_enter=True,
        min_lines=1,
        max_lines=5,
        filled=True,
        expand=True,
        on_submit=lambda e: send_message_click(e)
    )

    # Initialize Services Per Session
    # Use page.session_id or fallback to a random ID if not available
    session_id = page.session_id
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
    
    print(f"Initializing app for session: {session_id}")
    services = ServiceManager(session_id=session_id)
    
    # Track uploaded files
    uploaded_files_view = ft.Column()

    def add_message(text, sender="bot", sources=None):
        is_user = sender == "user"
        
        # Avatar
        avatar = ft.CircleAvatar(
            content=ft.Icon(ft.icons.PERSON if is_user else ft.icons.SMART_TOY),
            color=ft.colors.WHITE,
            bgcolor=ft.colors.BLUE if is_user else ft.colors.TEAL,
        )
        
        # Message Bubble
        bubble = ft.Container(
            content=ft.Markdown(
                text,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                code_theme="atom-one-dark",
            ),
            padding=15,
            border_radius=10,
            bgcolor=ft.colors.BLUE_GREY_900 if is_user else ft.colors.BLACK38,
            border=ft.border.all(1, ft.colors.WHITE10),
            expand=True
        )

        # Layout
        row = ft.Row(
            controls=[bubble, avatar] if is_user else [avatar, bubble],
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        
        chat_history.controls.append(row)
        
        
        # Meta info (Time/Status)
        meta_text = ft.Text("", size=10, color=ft.colors.GREY)

        # Add sources if available
        if sources:
            source_row = ft.Row(
                controls=[
                    ft.Container(width=40), # spacer for avatar
                    ft.Row(
                        [
                            ft.Row([ft.Chip(label=ft.Text(s), height=25) for s in sources], wrap=True),
                            meta_text
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        expand=True
                    )
                ]
            )
            chat_history.controls.append(source_row)
        elif not is_user:
             # If no sources but it's a bot, show timer
            meta_row = ft.Row(
                controls=[
                    ft.Container(width=40),
                    meta_text
                ]
            )
            chat_history.controls.append(meta_row)

        page.update()
        # Scroll to bottom
        chat_history.scroll_to(offset=-1, duration=300)
        
        return bubble.content, meta_text

    def process_query(query):
        if not query: 
            return

        add_message(query, "user")
        new_message.value = ""
        new_message.focus()
        page.update()

        # RAG Logic with Langfuse Tracking
        try:
            # Create a trace for this query
            trace_id = None
            if services.llm_service.langfuse.enabled:
                trace = services.llm_service.langfuse.create_trace(
                    name="rag_query",
                    user_id="flet_user",
                    metadata={
                        "query": query,
                        "model": services.llm_service.model,
                        "retrieval_k": settings.RETRIEVAL_K
                    },
                    input={"query": query}, # Using dict to be structured, or just query string
                    tags=["environment:development"]
                )
                trace_id = trace.id if trace else None
            
            results = services.vector_store.search(query, k=settings.RETRIEVAL_K, trace_id=trace_id)
            
            if not results:
                refusal_msg = random.choice(SOFT_REFUSAL_MESSAGES)
                add_message(refusal_msg, "bot")
                
                # Track empty results
                if trace_id and services.llm_service.langfuse.enabled:
                    services.llm_service.langfuse.track_span(
                        trace_id=trace_id,
                        name="no_results",
                        input_data={"query": query},
                        output_data={"message": refusal_msg}
                    )
                    # Update trace with refusal
                    services.llm_service.langfuse.update_trace(trace_id=trace_id, output=refusal_msg)

                return

            context_text = "\n\n".join([r.content for r in results])
            sources = list(set([r.source for r in results]))
            
            # Create initial message
            msg_control, meta_control = add_message("Thinking...", "bot", sources)
            
            # Start Timer
            start_time = time.time()
            
            # Stream response
            full_response = ""
            first_chunk = True
            
            for chunk in services.llm_service.generate_response_stream(context_text, query, trace_id=trace_id):
                if first_chunk:
                    full_response = ""
                    first_chunk = False
                
                full_response += chunk
                msg_control.value = full_response
                page.update()
            
            # Stop Timer
            elapsed = time.time() - start_time
            if elapsed < 60:
                time_str = f"{int(elapsed)}s"
            else:
                time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            
            meta_control.value = time_str
            
            # Final update to ensure everything is rendered
            page.update()
            
            # Flush Langfuse events
            if services.llm_service.langfuse.enabled:
                # Update the trace with the final response
                if trace_id:
                     services.llm_service.langfuse.update_trace(trace_id=trace_id, output=full_response)
                
                services.llm_service.langfuse.flush()
            
        except Exception as e:
            print(f"Query error: {e}")
            add_message(GENERIC_ERROR_MESSAGE, "bot")

    def send_message_click(e):
        process_query(new_message.value)

    # --- File Upload Logic ---
    def on_dialog_result(e):
        if not e.files:
            return
        
        for f in e.files:
            # For desktop Flet, path is available. 
            # For web, we need to handle upload (stream).
            # Assuming Desktop execution mostly based on user request "on top of desktop..."
            
            file_path = f.path
            filename = f.name
            if file_path:
                trace_id = None
                try:
                    print(f"Starting upload for: {filename}")
                    
                    # Create trace for document upload
                    try:
                        if services.llm_service.langfuse.enabled:
                            trace = services.llm_service.langfuse.create_trace(
                                name="document_upload",
                                user_id="flet_user",
                                metadata={
                                    "filename": filename,
                                    "file_path": file_path
                                }
                            )
                            trace_id = trace.id if trace else None
                            print(f"Langfuse trace created: {trace_id}")
                    except Exception as trace_error:
                        print(f"Langfuse trace creation failed (continuing): {trace_error}")
                    
                    upload_status.value = f"Processing {filename}..."
                    upload_status.color = ft.colors.BLUE
                    page.update()
                    
                    print(f"Processing file with doc_processor...")
                    chunks, page_count = services.doc_processor.process_file(file_path)
                    print(f"Created {len(chunks)} chunks from {page_count} pages")
                    
                    print(f"Adding documents to vector store...")
                    services.vector_store.add_documents(chunks, trace_id=trace_id)
                    print(f"Documents added successfully")
                    
                    # Add to UI list with delete button
                    add_file_to_list(filename)
                    
                    upload_status.value = f"✓ Indexed {filename} ({page_count} pages, {len(chunks)} chunks)"
                    upload_status.color = ft.colors.GREEN
                    
                    # Flush Langfuse events
                    try:
                        if services.llm_service.langfuse.enabled:
                            services.llm_service.langfuse.flush()
                    except Exception as flush_error:
                        print(f"Langfuse flush failed (ignoring): {flush_error}")
                    
                    print(f"Upload complete for: {filename}")
                        
                except Exception as ex:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Upload error for {filename}:")
                    print(error_details)
                    
                    # Show specific error to user
                    upload_status.value = f"✗ Error: {str(ex)[:50]}"
                    upload_status.color = ft.colors.RED
                    
                    # Show snackbar with error
                    page.snack_bar = ft.SnackBar(
                        ft.Text(f"Upload failed: {str(ex)[:100]}"),
                        bgcolor=ft.colors.RED_700,
                        open=True
                    )
                    
                    # Track error in Langfuse
                    try:
                        if trace_id and services.llm_service.langfuse.enabled:
                            services.llm_service.langfuse.track_span(
                                trace_id=trace_id,
                                name="upload_error",
                                input_data={"filename": filename},
                                output_data=None,
                                metadata={"error": str(ex)}
                            )
                    except Exception as lf_error:
                        print(f"Langfuse error tracking failed: {lf_error}")
            else:
                 upload_status.value = "Web upload not fully implemented in this local demo."
                 upload_status.color = ft.colors.ORANGE
                 
            page.update()

    def delete_file_click(e, filename, control_to_remove):
        try:
            services.vector_store.delete_document(filename)
            uploaded_files_view.controls.remove(control_to_remove)
            page.update()
        except Exception as ex:
            print(f"Delete error: {ex}")

    def add_file_to_list(filename):
        # Prevent duplicates in UI
        for c in uploaded_files_view.controls:
            if c.data == filename:
                return

        row = ft.Row(
            controls=[
                ft.Icon(ft.icons.INSERT_DRIVE_FILE, size=16),
                ft.Text(filename, size=12, expand=True, no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                ft.IconButton(
                    icon=ft.icons.DELETE_OUTLINE, 
                    icon_color=ft.colors.RED_200,
                    icon_size=16,
                    tooltip="Delete",
                    on_click=lambda e: delete_file_click(e, filename, row)
                )
            ],
            data=filename,
            spacing=5
        )
        uploaded_files_view.controls.append(row)
        page.update()

    file_picker = ft.FilePicker(on_result=on_dialog_result)
    page.overlay.append(file_picker)

    upload_btn = ft.ElevatedButton(
        "Upload Doc",
        icon=ft.icons.UPLOAD_FILE,
        on_click=lambda _: file_picker.pick_files(allow_multiple=True, allowed_extensions=["pdf", "txt"])
    )
    upload_status = ft.Text("Ready", size=12, color=ft.colors.GREY)


    # Layout Construction
    sidebar = ft.Container(
        width=250,
        padding=20,
        bgcolor=ft.colors.SURFACE_VARIANT,
        content=ft.Column([
            ft.Text("Knowledge Base", size=20, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            upload_btn,
            upload_status,
            ft.Container(height=10),
            ft.Text("Documents:", weight=ft.FontWeight.BOLD, size=14),
            uploaded_files_view,
            ft.Text("Documents:", weight=ft.FontWeight.BOLD, size=14),
            uploaded_files_view,
            ft.Divider(),
            ft.Text("Settings", weight=ft.FontWeight.BOLD),
            ft.Text("Model:", size=12),
            ft.Dropdown(
                width=200,
                options=[
                    ft.dropdown.Option("llama3.2:1b"),
                    ft.dropdown.Option("gpt-oss:120b-cloud"),
                    ft.dropdown.Option("mistral"),
                    ft.dropdown.Option("gemma2:2b"),
                ],
                value=settings.LLM_MODEL,
                on_change=lambda e: set_model(e.control.value),
                text_size=12,
                content_padding=5
            ),
            ft.Text(f"Embed: {settings.EMBEDDING_MODEL}", size=10, italic=True),
        ])
    ) 

    def set_model(model_name):
        try:
            services.llm_service.model = model_name
            print(f"Model switched to {model_name}")
            page.snack_bar = ft.SnackBar(ft.Text(f"Model switched to {model_name}"), open=True)
            page.update()
        except Exception as e:
            print(f"Error switching model: {e}")

    chat_input_container = ft.Container(
        padding=20,
        bgcolor=ft.colors.BACKGROUND,
        content=ft.Row([
            new_message,
            ft.IconButton(icon=ft.icons.SEND_ROUNDED, icon_size=30, on_click=send_message_click)
        ])
    )

    layout = ft.Row(
        [
            sidebar,
            ft.VerticalDivider(width=1),
            ft.Column([
                ft.Container(
                    content=chat_history,
                    expand=True,
                    padding=20,
                ),
                chat_input_container
            ], expand=True)
        ],
        expand=True
    )

    page.add(layout)
    
    # Welcome message
    add_message("Welcome! Upload a document to get started.", "bot")

if __name__ == "__main__":
    ft.app(target=main)
