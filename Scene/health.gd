extends Node
class_name Health

signal damaged(current_health: float, max_health: float)
signal died()

@export var max_health: float = 100

var current_health: float
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	current_health = max_health
	
func damage(damage_amount:float) -> void:
	current_health = max(0,current_health - damage_amount)
	emit_signal("damaged",current_health, max_health)
	
	if current_health == 0:
		emit_signal("died")
