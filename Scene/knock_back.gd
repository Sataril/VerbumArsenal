extends Node2D
class_name KnockBack

signal start_knock_back(direction: Vector2, force: float, duration: float)
signal stop_knock_back()
@onready var timer: Timer = $Timer

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	timer.timeout.connect(on_time_out)

func apply_knock_back(direction: Vector2, force: float, duration: float) -> void:
	timer.wait_time = duration
	timer.start()
	emit_signal("start_knock_back",direction, force, duration)
	
func on_time_out()-> void:
	emit_signal("stop_knock_back")
	
