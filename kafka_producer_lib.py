from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
import json

class KafkaProducerLib:
    def __init__(self, bootstrap_servers, topic_name):
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.producer = Producer(
            {
                'bootstrap.servers': self.bootstrap_servers
                })
        self._create_topic()

    def _create_topic(self):
        """Create a topic if it doesn't exist."""
        admin_client = AdminClient({'bootstrap.servers': self.bootstrap_servers})
        topic = NewTopic(self.topic_name, num_partitions=3, replication_factor=1)
        
        # Create the topic
        fs = admin_client.create_topics([topic])
        for topic, future in fs.items():
            try:
                future.result()
                print(f"Topic '{topic}' created successfully!")
            except Exception as e:
                print(f"Failed to create topic '{topic}': {e}")

    def send_message(self, message):
        """Send message to Kafka topic."""
        try:
            self.producer.produce(self.topic_name, value=json.dumps(message), callback=self.delivery_report)
            self.producer.flush()
            print(f"Message sent to {self.topic_name}")
        except Exception as e:
            print(f"Error sending message: {e}")
    
    def delivery_report(self, err, msg):
        """Callback function to report delivery status."""
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            print(f"Message delivered to {msg.topic()} [{msg.partition}] @ offset {msg.offset()}")

# Example usage
if __name__ == "__main__":
    producer_lib = KafkaProducerLib(bootstrap_servers='localhost:9092', topic_name='my-new-topic')
    
    # Example message to send
    message = {"transaction_id": "12345", "amount": 100.0, "status": "approved"}
    
    # Send the message
    producer_lib.send_message(message)
