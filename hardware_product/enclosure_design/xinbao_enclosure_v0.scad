// Xinbao desktop robot enclosure v1
// Parametric OpenSCAD source for the first engineering-validation print.
//
// Export one part at a time:
//   openscad -o stl/xinbao_front_shell_v1.stl -D PART_ID=1 xinbao_enclosure_v0.scad
//   openscad -o stl/xinbao_back_shell_v1.stl  -D PART_ID=2 xinbao_enclosure_v0.scad
//   openscad -o stl/xinbao_base_v1.stl        -D PART_ID=3 xinbao_enclosure_v0.scad

$fn = 72;

// Choose exported part:
//   0 = assembly preview
//   1 = front shell
//   2 = back shell
//   3 = base
PART_ID = 0;

// Main body envelope, mm
body_w = 175;
body_d = 125;
body_h = 190;
wall = 3;
corner_r = 16;
front_depth = 52;
base_h = 18;

// Screen default: suitable for many 5 inch HDMI modules.
// Measure your actual screen and update these before final printing.
screen_window_w = 112;
screen_window_h = 68;
screen_center_z = 118;
screen_mount_w = 126;
screen_mount_h = 82;
screen_screw_d = 2.8;

// M260C dimensions from the local STEP file:
// about 72.9 x 75.75 x 3.25 mm. The recess is intentionally loose.
mic_outer_d = 82;
mic_recess_depth = 1.6;
mic_hole_d = 5;
mic_hole_r = 30;
mic_center_hole_d = 22;
mic_y = -20;

// Audio output: one 40 mm speaker for v0.
speaker_d = 40;
speaker_z = 43;
speaker_grille_hole_d = 3.0;

// Back fan and service opening.
fan_d = 40;
fan_center_z = 134;
fan_screw_spacing = 32;
fan_screw_d = 3.2;
port_window_w = 92;
port_window_h = 34;
port_window_z = 78;

// Orange Pi AI Pro 20T official case from the provided image is about
// 124.2 x 88.6 x 36 mm.
// This v1 assumes the board stays inside the official case, so the base
// uses a loose tray instead of bare-board screw standoffs.
pi_case_w = 124.2;
pi_case_d = 88.6;
pi_case_h = 36;
pi_tray_clearance = 2;
tray_wall = 4;
tray_h = 10;

seam_y = -body_d / 2 + front_depth;

module rounded_box(size, r) {
  linear_extrude(height = size[2])
    offset(r = r)
      square([size[0] - 2 * r, size[1] - 2 * r], center = true);
}

module body_outer() {
  rounded_box([body_w, body_d, body_h], corner_r);
}

module hollow_body_shell() {
  difference() {
    body_outer();
    translate([0, 0, -1])
      rounded_box(
        [body_w - 2 * wall, body_d - 2 * wall, body_h - wall + 1],
        max(1, corner_r - wall)
      );
  }
}

module front_clip() {
  translate([-body_w, -body_d / 2 - 2, -2])
    cube([2 * body_w, front_depth + 4, body_h + 4]);
}

module back_clip() {
  translate([-body_w, seam_y - 2, -2])
    cube([2 * body_w, body_d, body_h + 4]);
}

module y_cylinder(d, h) {
  rotate([90, 0, 0]) cylinder(d = d, h = h, center = true);
}

module x_cylinder(d, h) {
  rotate([0, 90, 0]) cylinder(d = d, h = h, center = true);
}

module screen_mount_bosses() {
  y = -body_d / 2 + wall + 5;
  for (x = [-screen_mount_w / 2, screen_mount_w / 2])
    for (z = [screen_center_z - screen_mount_h / 2, screen_center_z + screen_mount_h / 2])
      translate([x, y, z])
        y_cylinder(d = 10, h = 10);
}

module screen_screw_holes() {
  y = -body_d / 2 + wall + 5;
  for (x = [-screen_mount_w / 2, screen_mount_w / 2])
    for (z = [screen_center_z - screen_mount_h / 2, screen_center_z + screen_mount_h / 2])
      translate([x, y, z])
        y_cylinder(d = screen_screw_d, h = 24);
}

module speaker_support_ring() {
  y = -body_d / 2 + wall + 3;
  difference() {
    translate([0, y, speaker_z])
      y_cylinder(d = speaker_d + 8, h = 6);
    translate([0, y, speaker_z])
      y_cylinder(d = speaker_d + 1, h = 8);
  }
}

module speaker_grille_cutouts() {
  for (x = [-18 : 6 : 18])
    for (z = [-18 : 6 : 18])
      if (sqrt(x * x + z * z) < speaker_d / 2 - 1)
        translate([x, -body_d / 2 - 1, speaker_z + z])
          y_cylinder(d = speaker_grille_hole_d, h = wall + 8);
}

module mic_top_cutouts() {
  translate([0, mic_y, body_h - mic_recess_depth])
    cylinder(d = mic_outer_d, h = mic_recess_depth + 2);

  translate([0, mic_y, body_h - wall - 2])
    cylinder(d = mic_center_hole_d, h = wall + 6);

  for (a = [0 : 60 : 300])
    translate([mic_hole_r * cos(a), mic_y + mic_hole_r * sin(a), body_h - wall - 2])
      cylinder(d = mic_hole_d, h = wall + 6);
}

module side_vent_cutouts() {
  for (side = [-1, 1])
    for (z = [82 : 10 : 152])
      translate([side * (body_w / 2 - 2), 8, z])
        cube([18, 58, 3.2], center = true);
}

module front_cutouts() {
  translate([-screen_window_w / 2, -body_d / 2 - 2, screen_center_z - screen_window_h / 2])
    cube([screen_window_w, wall + 8, screen_window_h]);

  screen_screw_holes();
  speaker_grille_cutouts();
  mic_top_cutouts();
  side_vent_cutouts();
}

module front_shell() {
  difference() {
    union() {
      intersection() {
        hollow_body_shell();
        front_clip();
      }
      screen_mount_bosses();
      speaker_support_ring();
    }
    front_cutouts();
  }
}

module fan_grille_cutouts() {
  for (x = [-15 : 6 : 15])
    for (z = [-15 : 6 : 15])
      if (sqrt(x * x + z * z) < fan_d / 2 - 2)
        translate([x, body_d / 2 + 1, fan_center_z + z])
          y_cylinder(d = 3.2, h = wall + 8);

  for (x = [-fan_screw_spacing / 2, fan_screw_spacing / 2])
    for (z = [fan_center_z - fan_screw_spacing / 2, fan_center_z + fan_screw_spacing / 2])
      translate([x, body_d / 2 + 1, z])
        y_cylinder(d = fan_screw_d, h = wall + 8);
}

module back_cutouts() {
  translate([-port_window_w / 2, body_d / 2 - wall - 2, port_window_z - port_window_h / 2])
    cube([port_window_w, wall + 8, port_window_h]);

  fan_grille_cutouts();
  side_vent_cutouts();
}

module back_shell() {
  difference() {
    intersection() {
      hollow_body_shell();
      back_clip();
    }
    back_cutouts();
    mic_top_cutouts();
  }
}

module case_tray() {
  tray_w = pi_case_w + 2 * pi_tray_clearance;
  tray_d = pi_case_d + 2 * pi_tray_clearance;

  difference() {
    translate([0, 0, base_h])
      rounded_box([tray_w + 2 * tray_wall, tray_d + 2 * tray_wall, tray_h], 8);
    translate([0, 0, base_h - 1])
      rounded_box([tray_w, tray_d, tray_h + 3], 5);
    // Rear cable relief for the current Type-C power cable and HDMI/USB access.
    translate([-42, tray_d / 2 - 4, base_h + 1])
      cube([84, 22, tray_h + 3]);
  }
}

module shell_screw_hole(x, y) {
  translate([x, y, -1])
    cylinder(d = 3.2, h = base_h + 4);
}

module base_plate() {
  difference() {
    union() {
      rounded_box([body_w + 10, body_d + 10, base_h], corner_r + 4);
      case_tray();
    }

    // Cable exit at the rear.
    translate([-18, body_d / 2 + 2, 5])
      cube([36, 18, 10]);

    // Shell fastening holes.
    for (x = [-body_w / 2 + 18, body_w / 2 - 18])
      for (y = [-body_d / 2 + 18, body_d / 2 - 18])
        shell_screw_hole(x, y);

    // Lightening and bottom airflow openings.
    for (x = [-45, -15, 15, 45])
      translate([x, 0, -1])
        cube([12, 74, base_h + 3], center = false);
  }
}

module assembly_preview() {
  color("#f3f3f3") front_shell();
  color("#d8e5ef") back_shell();
  color("#20242b") translate([0, 0, -base_h]) base_plate();

  // Hardware envelope previews only; not exported as printable parts.
  color("#ff4d7d", 0.35)
    translate([-screen_window_w / 2, -body_d / 2 - 1, screen_center_z - screen_window_h / 2])
      cube([screen_window_w, 3, screen_window_h]);

  color("#43d9ff", 0.35)
    translate([-board_w / 2, -board_d / 2, -base_h + base_h + standoff_h])
      cube([board_w, board_d, 2]);
}

if (PART_ID == 1) {
  front_shell();
} else if (PART_ID == 2) {
  back_shell();
} else if (PART_ID == 3) {
  base_plate();
} else {
  assembly_preview();
}
