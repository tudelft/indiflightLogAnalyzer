/*
    Quadrotor visuals with THREE.js and data from a websocket connection

    Copyright (C) 2024 Till Blaha -- TU Delft

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ViewHelper } from "three/addons/helpers/ViewHelper.js";
import Stats from "stats.js";
import {
    craftTypes,
    quadRotor,
    tailSitter,
    airplane,
    } from "./crafts.js";

var clock = new THREE.Clock();

function updateVisualization(data) {
    for (let i = 0; i < data.length; i++) {
        if (!idList.includes(data[i].id)) {
            // never seen this id, add new craft to scene
            var newCraft;
            switch (data[i].type) {
                case craftTypes.quadRotor:
                    newCraft = new quadRotor(0.12, 0.09, 3.*0.0254);
                    break;
                case craftTypes.tailSitter:
                    newCraft = new tailSitter(1., 0.3, 6*0.0254);
                    break;
                case craftTypes.airplane:
                    newCraft = new airplane(1., 0.3, 8*0.0254);
                    break;
                default:
                    break;
            }
            craftList.push(newCraft);
            newCraft.addToScene(scene);
            idList.push(data[i].id);
        }
        var idx = idList.indexOf(data[i].id);
        craftList[idx].setPose(data[i].pos, data[i].quat);
        craftList[idx].setControls(data[i].ctl);
    }
}
window.updateVisualization = updateVisualization;

function fetchPose() {
    fetch('/pose')
    //fetch('/pose_showoff')
      .then(response => {
        // Check if the request was successful (status code 200)
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        // Parse the response as JSON
        return response.json();
      })
      .then(data => {
        // Use the parsed JSON data
        updateVisualization(data);
      })
      .catch(error => {
        // Handle errors
        console.error('There was a problem with the fetch operation:', error);
      });
}

var time_since_fetch = 0;

function animate() {
    stats.begin()

	requestAnimationFrame( animate );

    const delta = clock.getDelta();
    time_since_fetch += delta;
    if (time_since_fetch > 0.03) {
        time_since_fetch = 0;
        fetchPose();
    }
    if ( viewHelper.animating ) viewHelper.update( delta );

    renderer.clear();
	renderer.render( scene, camera );
    viewHelper.render( renderer );

    stats.end()
}

var idList = []
var craftList = []
window.craftList = craftList

// webGL renderer
const renderer = new THREE.WebGLRenderer( { antialias: true } );
renderer.setSize( window.innerWidth, window.innerHeight );
renderer.autoClear = false;
document.body.appendChild( renderer.domElement )

// scene with white background
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

// camera, such that North East Down makes sense
const camera = new THREE.PerspectiveCamera( 40, window.innerWidth / window.innerHeight, 0.1, 1000 );
camera.up.set( 0, 0, -1 ); // for orbit controls to make sense
camera.position.x = -0.8;
camera.position.y = 0.7;
camera.position.z = -0.45;
//camera.setRotationFromEuler( new THREE.Euler(-140*3.1415/180, 0, 85 * 3.1415/180, 'ZYX'))
camera.setRotationFromEuler( new THREE.Euler(-115*3.1415/180, 0, 55 * 3.1415/180, 'ZYX'))

window.onresize = function() {
    var margin = 35;
    camera.aspect = (window.innerWidth-margin) / (window.innerHeight-margin);
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth-margin, window.innerHeight-margin);
};

// ground plane grid in the xy-plane and coordinate system stems
const gd = new THREE.GridHelper( 1, 4 );
gd.rotation.x = -0.5*3.1415
//scene.add( gd );
var axisHelper2 = new THREE.AxesHelper ( 0.25 )
axisHelper2.setColors(0x00000, 0x00000, 0x00000)
scene.add( axisHelper2 );


const imuGeometry = new THREE.BufferGeometry();
const imuVertices = new Float32Array ( [
            -0.01, 0.01, 0,  // Top
            0.01, 0.01, 0,  // Top
            0.01, -0.01, 0,  // Top
            -0.01, -0.01, 0,  // Top
] );
const imuIndices = [
    0, 1, 2,
    2, 3, 0,
    0, 2, 1,
    2, 0, 3
];

imuGeometry.setIndex( imuIndices );
imuGeometry.setAttribute( 'position', new THREE.BufferAttribute( imuVertices, 3 ) );

const darkBlueMaterial = new THREE.MeshBasicMaterial({ color: 0x000064 });
const imuMesh = new THREE.Mesh(imuGeometry, darkBlueMaterial);

const imuObject = new THREE.Object3D();
imuObject.add(imuMesh);
const axisHelper = new THREE.AxesHelper (0.15);
axisHelper.setColors(0x00000, 0x00000, 0x00000)
imuObject.add( axisHelper );
imuObject.position.x = 0.16
imuObject.position.y = 0.1
imuObject.position.z = -0.16
imuObject.rotation.x = 0.25
imuObject.rotation.z = 0.0

scene.add(imuObject);

var vector = new THREE.Vector3( 0.16, 0.1, -0.16 );
var vectorNorm = new THREE.Vector3( 0.16, 0.1, -0.16 ).normalize();
var vlen = vector.length()

var arrow = new THREE.ArrowHelper(
    vectorNorm,
    new THREE.Vector3( 0., 0., 0. ),
    vlen,
    0x606060,
    );

arrow.setLength(
    vlen,
    0.03,
    0.015,
)

//scene.add(arrow)


var pos = new THREE.Vector3( 0.21, 0.1, -0.21 );
//var posVector = new THREE.Vector3( pos.x - vector.x, pos.y - vector.y, pos.z - vector.z )
var posVector = new THREE.Vector3( vector.x - pos.x, vector.y - pos.y, vector.z - pos.z )
var posvlen = posVector.length()

var arrow = new THREE.ArrowHelper(
    posVector.normalize(),
    //vector,
    pos,
    posvlen,
    0x606060,
    );

arrow.setLength(
    posvlen,
    0.03,
    0.015,
)

scene.add(arrow)


// interactive camera controls and triad in the corner
//const controls = new OrbitControls( camera, renderer.domElement );
//document.addEventListener('keydown', function(event) { // reset view on space
//    if (event.code === 'Space') { controls.reset(); } });
var viewHelper = new ViewHelper( camera, renderer.domElement );
//viewHelper.controls = controls;
//viewHelper.controls.center = controls.target;
//window.onpointerup = function (event) { // enable clicking of the triad
//    viewHelper.handleClick( event ) };

window.onresize() // call once

// performance counter in top left
const stats = new Stats()
stats.showPanel(0) // 0: fps, 1: ms, 2: mb, 3+: custom
document.body.appendChild(stats.dom)

animate();
