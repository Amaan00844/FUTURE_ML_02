"""
Demo Script: Quick Ticket Prediction
Shows how to use the trained model for real-time predictions
"""

from ticket_classifier import TicketClassifier

def demo_predictions():
    """Demonstrate ticket classification with various examples"""
    
    print("="*70)
    print("SUPPORT TICKET CLASSIFICATION - DEMO")
    print("="*70)
    
    # Initialize and prepare the classifier
    print("\n⏳ Loading and preparing the model...")
    classifier = TicketClassifier('all_tickets_processed_improved_v3.csv')
    classifier.load_data()
    classifier.prepare_features()
    _, best_model = classifier.train_models()
    
    print("\n✅ Model ready for predictions!\n")
    
    # Demo tickets
    demo_tickets = [
        {
            "text": "My laptop won't connect to the WiFi network. Urgent help needed!",
            "description": "Connectivity Issue"
        },
        {
            "text": "Need approval for purchasing new monitors for the development team",
            "description": "Purchase Request"
        },
        {
            "text": "Cannot access the employee portal to submit my timesheet",
            "description": "Portal Access Issue"
        },
        {
            "text": "Running out of storage space on the shared drive",
            "description": "Storage Issue"
        },
        {
            "text": "Need administrative rights to install development software",
            "description": "Permission Request"
        },
        {
            "text": "Question about the new vacation policy",
            "description": "HR Query"
        },
        {
            "text": "Project deadline approaching, need additional resources",
            "description": "Internal Project"
        },
        {
            "text": "Printer in conference room B is not working",
            "description": "Hardware Issue"
        }
    ]
    
    # Process each ticket
    print("="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    for i, ticket in enumerate(demo_tickets, 1):
        print(f"\n{'─'*70}")
        print(f"TICKET #{i}: {ticket['description']}")
        print(f"{'─'*70}")
        print(f"Text: \"{ticket['text']}\"")
        print()
        
        # Get prediction
        result = classifier.predict_new_ticket(ticket['text'])
        
        # Display results
        print(f"🎯 PREDICTED CATEGORY: {result['category']}")
        print(f"⚡ PRIORITY LEVEL: {result['priority']}")
        
        if result['confidence']:
            print(f"📊 CONFIDENCE: {result['confidence']:.1%}")
            
            print(f"\n   Top 3 Predictions:")
            for j, (category, prob) in enumerate(result['top_predictions'][:3], 1):
                bar_length = int(prob * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                print(f"   {j}. {category:25s} {bar} {prob:.1%}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    
    # Provide usage instructions
    print("\n💡 TO USE IN YOUR CODE:")
    print("""
    from ticket_classifier import TicketClassifier
    
    classifier = TicketClassifier('dataset.csv')
    classifier.load_data()
    classifier.prepare_features()
    classifier.train_models()
    
    result = classifier.predict_new_ticket("Your ticket text here")
    print(f"Category: {result['category']}")
    print(f"Priority: {result['priority']}")
    """)

if __name__ == "__main__":
    demo_predictions()
