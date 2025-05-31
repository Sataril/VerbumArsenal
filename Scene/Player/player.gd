extends CharacterBody2D
class_name Player
@export var move_speed: int = 100
@onready var animated_sprite_2d: AnimatedSprite2D = $AnimatedSprite2D
@onready var knock_back: Node2D = $KnockBack
@onready var health: Health = $Health

var is_knocking_back: bool = false

func _ready() ->void:
	knock_back.start_knock_back.connect(on_knock_back_started)
	knock_back.stop_knock_back.connect(on_knock_back_stopped)
	
	health.died.connect(on_died)
	
func _process (delta: float) -> void:
	if is_knocking_back:
		move_and_slide()
		return
	var move_direction = Input.get_vector("move_left","move_right","move_up","move_down")
	velocity = move_direction * move_speed
	handle_animation(move_direction)
	handle_location(move_direction)
	move_and_slide()	
	
func handle_animation(move_direction: Vector2) -> void:
	if move_direction.length() > 0:
		animated_sprite_2d.play("run")
	else:
		animated_sprite_2d.play("idle")

func handle_location(move_direction: Vector2) -> void:
	if move_direction.x > 0:
		animated_sprite_2d.scale = Vector2(1,1)
	elif move_direction.x < 0:
		animated_sprite_2d.scale = Vector2(-1,1)

func on_knock_back_started(direction: Vector2, force: float, duration: float) -> void:
	is_knocking_back = true
	velocity = direction *force

func on_knock_back_stopped() -> void:
	is_knocking_back = false
	velocity = Vector2.ZERO

func on_died() -> void:
	get_tree().call_deferred("reload_current_scene")
