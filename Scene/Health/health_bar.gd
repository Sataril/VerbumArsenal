extends ProgressBar
@export var health: Health
@export var is_player_health_bar: bool = false

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	if is_player_health_bar:
		health = get_tree().current_scene.get_node("Player").get_node_or_null("Health")
	if health != null:
		call_deferred("initialize")
		
func initialize()-> void:
	max_value = health.max_health
	min_value =0
	value = health.current_health
	health.damaged.connect(on_damaged)
	
func on_damaged(current_health: float, max_health: float)-> void:
	max_value = max_health
	value = current_health
