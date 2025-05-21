extends Node2D

@export var projectile_scene: PackedScene

@onready var shoot_timer: Timer = $ShootTimer

var start_shooting: bool = false 
var ready_to_shoot: bool = true

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("shoot"):
		start_shooting = true 
	if event.is_action_released("shoot"):
		start_shooting = false
		

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	shoot_timer.timeout.connect(on_time_out)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	handle_shoot()

func on_time_out() -> void:
	ready_to_shoot = true

func handle_shoot() -> void:
	if not start_shooting: 
		return
	if not ready_to_shoot:
		return

	var projectile: Projectiles = projectile_scene.instantiate() as Projectiles
	get_tree().current_scene.add_child(projectile)
	
	var mouse_global_position = get_global_mouse_position()
	var shoot_direction = mouse_global_position - global_position
	
	projectile.set_up(shoot_direction)
	projectile.global_position = global_position
	
	shoot_timer.start()
	ready_to_shoot = false
	
