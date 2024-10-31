from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch

def train_model(train_loader, batch_size=32, num_epochs=3):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Print the device being used

    # Load the model and move it to the appropriate device
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    if torch.cuda.device_count() > 1: # Wrap the model for multi-GPU training
      device_ids = list(range(torch.cuda.device_count()))  # Use all available GPUs
      model = torch.nn.DataParallel(model, device_ids=device_ids)
      # model = torch.nn.DataParallel(model)
      print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print("Using single GPU or CPU")

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Print GPU status at the beginning of each epoch
        # print_gpu_status()

        for batch in train_loader:
            # Move batch data to the appropriate device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output_ids = batch['output_ids'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids)
            loss = outputs.loss # Single GPU
            # loss = outputs.loss.mean()  # Compute mean to ensure a single scalar loss (For multiple GPU)

            # Print the current GPU being used
            # print(f"Current GPU: {torch.cuda.current_device()}")  # This will print the ID of the current GPU

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

    # Save the trained model
    model.save_pretrained('data/full_12m_sentences/t5_model_3_epochs')
    # model.module.save_pretrained('data/500k_sentences_suffled/t5_model_mul_gpu_8_num_workers')

    return model
