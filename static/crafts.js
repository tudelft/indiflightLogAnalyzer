import * as THREE from "three";

window.THREE = THREE

export const craftTypes = {
    quadRotor: 0,
    tailSitter: 2,
    airplane: 1,
};

export class quadRotor {
    constructor(width, length, diameter) {
        this.width = width
        this.length = length
        this.diameter = diameter
        const xs = [-length/2, length/2, -length/2, length/2];
        const ys = [width/2, width/2, -width/2, -width/2];
        const rotorGeo = new THREE.CircleGeometry( .5*this.diameter, 16 );
        const edges = new THREE.EdgesGeometry( rotorGeo ); 
        const mat = new THREE.LineBasicMaterial( { color: 0x000000 } );

        const triangleGeometry = new THREE.BufferGeometry();
        const vertices = new Float32Array ( [
                    length*0.5, 0, 0,  // Top
                    0, -width/6, 0, // Bottom left
                    0, +width/6, 0,   // Bottom right
        ] );
        const indices = [
            0,1,2, // top
            0,2,1, // bottom (right hand rule)
        ];

        triangleGeometry.setIndex( indices );
        triangleGeometry.setAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );

        const darkGreenMaterial = new THREE.MeshBasicMaterial({ color: 0x006400 });
        const triangleMesh = new THREE.Mesh(triangleGeometry, darkGreenMaterial);

        this.obj = new THREE.Object3D();
        //this.obj.add(triangleMesh);

        var rotorLines = [];
        this.thrustArrows = [];
        for (let i = 0; i < 4; i++) {
            var line = new THREE.LineSegments(edges, mat);
            this.obj.add(line);
            rotorLines.push(line);
            rotorLines[i].position.x = xs[i];
            rotorLines[i].position.y = ys[i];
            //var arrow = new THREE.ArrowHelper(
            //    new THREE.Vector3( 0., 0., -1. ),
            //    new THREE.Vector3( xs[i], ys[i], 0. ),
            //    0.,
            //    0xaa5500,
            //    );
            //this.obj.add(arrow);
            //this.thrustArrows.push(arrow);
        }
        let axisHelper = new THREE.AxesHelper ( 0.15 )
        axisHelper.setColors(0x00000, 0x00000, 0x00000)
        this.obj.add( axisHelper );

        const geometry = new THREE.SphereGeometry( 0.005, 32, 16 ); 
        const material = new THREE.MeshBasicMaterial( { color: 0x404040 } ); 
        const sphere = new THREE.Mesh( geometry, material );
        this.obj.add(sphere)
    }
    addToScene(theScene) {
        theScene.add(this.obj);
    }
    setPose(pos, quat) {
        this.obj.position.x = pos[0];
        this.obj.position.y = pos[1];
        this.obj.position.z = pos[2];
        this.obj.quaternion.w = quat[0];
        this.obj.quaternion.x = quat[1];
        this.obj.quaternion.y = quat[2];
        this.obj.quaternion.z = quat[3];
    }
    setControls(controls) {
        //for (let i = 0; i < controls.length; i++) {
        //    if (i >= this.thrustArrows.length)
        //        break;

        //    let arrowLength = 2.*controls[i]*this.diameter
        //    let arrowHeadLength = arrowLength < 0.4*this.diameter ? arrowLength : 0.4*this.diameter;
        //    this.thrustArrows[i].setLength(
        //        arrowLength,
        //        arrowHeadLength,
        //        0.4*this.diameter,
        //        );
        //}
    }
}


export class tailSitter {
    constructor(width, length, diameter) {
        this.width = width
        this.length = length
        this.diameter = diameter
        this.nRotors = 2
        this.nFlaps = 2
        const zs = [-length, -length];
        const ys = [-width/4, width/4];
        const rotorGeo = new THREE.CircleGeometry( .5*diameter, 16 );
        const edges = new THREE.EdgesGeometry( rotorGeo ); 
        const mat = new THREE.LineBasicMaterial( { color: 0x000000 } );

        const bodyGeometry = new THREE.BufferGeometry();
        const flapWidth = width/2 - width/8;
        const flapLength = length/4;
        const vertices = new Float32Array ( [
                    0, 0, -length,  // Top
                    0, -width/2, -length*0.6, // left top
                    0, -width/2, -flapLength,   // left bottom
                    0, -width/2+flapWidth, -flapLength,
                    0, -width/2+flapWidth, 0,
                    0, +width/2-flapWidth, 0,
                    0, +width/2-flapWidth, -flapLength,
                    0, +width/2, -flapLength,
                    0, +width/2, -length*0.6,   // right top
        ] );
        const indices = [
            0,1,2, // top
            0,2,3,
            0,3,4,
            0,4,5,
            0,5,6,
            0,6,7,
            0,7,8,
            //0,4,3, // bottom (right hand rule). Not needed for Phong with THREE.DoubleSide
        ];

        bodyGeometry.setIndex( indices );
        bodyGeometry.setAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );

        const transparentBodyMat = new THREE.MeshPhongMaterial({ color: 0x006400, opacity:0.6, transparent: true, side: THREE.DoubleSide });
        const triangleMesh = new THREE.Mesh(bodyGeometry, transparentBodyMat);

        this.obj = new THREE.Object3D();
        this.obj.add(triangleMesh);

        this.flaps = [];
        const flapVertices = new Float32Array ( [
            0, 0, 0,
            0, flapWidth, 0,
            0, flapWidth, flapLength,
            0, 0, flapLength,
        ] );

        const flapIndices = [
            0,1,2,
            0,2,3,
        ];

        var flapGeometry = new THREE.BufferGeometry();
        flapGeometry.setIndex( flapIndices );
        flapGeometry.setAttribute( 'position', new THREE.BufferAttribute( flapVertices, 3 ) );
        var darkGreenMat = new THREE.MeshBasicMaterial({ color: 0x006400, side: THREE.DoubleSide });

        for (let i = 0; i < 2; i++) {
            var flapMesh = new THREE.Mesh(flapGeometry, darkGreenMat);
            flapMesh.position.z = -flapLength;
            switch (i) {
                case 0:
                    flapMesh.position.y = -width/2;
                    break;
                case 1:
                    flapMesh.position.y = width/2 - flapWidth;
                    break;
            }
            this.flaps.push(flapMesh);
            this.obj.add(flapMesh);
        }

        var rotorLines = [];
        this.thrustArrows = [];
        for (let i = 0; i < ys.length; i++) {
            var line = new THREE.LineSegments(edges, mat);
            this.obj.add(line);
            rotorLines.push(line);
            rotorLines[i].position.y = ys[i];
            rotorLines[i].position.z = zs[i];
            var arrow = new THREE.ArrowHelper(
                new THREE.Vector3( 0., 0., -1. ),
                new THREE.Vector3( 0., ys[i], zs[i] ),
                0.,
                0xaa5500,
                );
            this.obj.add(arrow);
            this.thrustArrows.push(arrow);
        }
    }
    addToScene(theScene) {
        theScene.add(this.obj);
    }
    setPose(pos, quat) {
        this.obj.position.x = pos[0];
        this.obj.position.y = pos[1];
        this.obj.position.z = pos[2];
        this.obj.quaternion.w = quat[0];
        this.obj.quaternion.x = quat[1];
        this.obj.quaternion.y = quat[2];
        this.obj.quaternion.z = quat[3];
    }
    setControls(controls) {
        for (let i = 0; i < this.nRotors; i++) {
            if (i >= controls.length)
                break;

            let arrowLength = 2.*controls[i]*this.diameter
            let arrowHeadLength = arrowLength < 0.4*this.diameter ? arrowLength : 0.4*this.diameter;
            this.thrustArrows[i].setLength(
                arrowLength,
                arrowHeadLength,
                0.4*this.diameter,
                );
        }
        for (let i = 0; i < this.nFlaps; i++) {
            var idx = i + this.nRotors;
            if (idx >= controls.length)
                break;

            this.flaps[i].rotation.y = controls[idx]*0.8;
        }
    }
}

export class airplane {
    constructor(width, length, diameter) {
        this.width = width
        this.length = length
        this.diameter = diameter
        this.nRotors = 1
        this.nFlaps = 5
        const xs = [length*1.3];
        const ys = [0];
        const rotorGeo = new THREE.CircleGeometry( .5*diameter, 16 );
        const edges = new THREE.EdgesGeometry( rotorGeo ); 
        const mat = new THREE.LineBasicMaterial( { color: 0x000000 } );

        const ailWidth = width/2 - width/8;
        const ailLength = length/4;
        const vertices = new Float32Array ( [
                    length, 0, 0,  // Top
                    length, -width/2, 0, // left top
                    ailLength, -width/2, 0,
                    ailLength, -width/2+ailWidth, 0,
                    -0, -width/2+ailWidth, 0,
                    -0, +width/2-ailWidth, 0,
                    ailLength, +width/2-ailWidth, 0,
                    ailLength, +width/2, 0,
                    length, +width/2, 0,
        ] );
        const indices = [
            0,1,2, // top
            0,2,3,
            0,3,4,
            0,4,5,
            0,5,6,
            0,6,7,
            0,7,8,
            //0,4,3, // bottom (right hand rule). Not needed for Phong with THREE.DoubleSide
        ];

        const bodyGeometry = new THREE.BufferGeometry();
        bodyGeometry.setIndex( indices );
        bodyGeometry.setAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );

        const horzGeometry = new THREE.BufferGeometry();
        horzGeometry.setIndex( indices );
        horzGeometry.setAttribute( 'position', new THREE.BufferAttribute( vertices.map(value => value*0.5), 3 ) );

        const vertGeometry = new THREE.BufferGeometry();
        const vertVertices = new Float32Array ( [
                    -length*1.85, 0, -length*0.8,
                    -length*1.85, 0, 0,
                    0, 0, 0,
                    0, 0, -length/8,
                    -length*1.4, 0, -length/8,
                    -length*1.6, 0, -length*0.8,
        ] );
        const vertIndices = [
            5, 0, 1,
            1, 4, 5,
            1, 2, 4, 
            2, 3, 4,
        ]
        vertGeometry.setIndex( vertIndices );
        vertGeometry.setAttribute( 'position', new THREE.BufferAttribute( vertVertices, 3 ) );

        const transparentBodyMat = new THREE.MeshPhongMaterial({ color: 0x006400, opacity:0.6, transparent: true, side: THREE.DoubleSide });
        const wingMesh = new THREE.Mesh(bodyGeometry, transparentBodyMat);
        const horzMesh = new THREE.Mesh(horzGeometry, transparentBodyMat);
        horzMesh.position.x = -length*2
        const vertMesh = new THREE.Mesh(vertGeometry, transparentBodyMat);

        this.obj = new THREE.Object3D();
        this.obj.add(wingMesh);
        this.obj.add(horzMesh);
        this.obj.add(vertMesh);

        const ailVertices = new Float32Array ( [
            0, 0, 0,
            0, ailWidth, 0,
            -ailLength, ailWidth, 0,
            -ailLength, 0, 0,
        ] );

        const ailIndices = [
            0,1,2,
            0,2,3,
        ];

        const rudVertices = new Float32Array ( [
            0, 0, 0,
            -0.15*length, 0, 0,
            -0.15*length, 0, -0.8*length,
            0, 0, -0.8*length,
        ] );

        const rudIndices = [
            0,1,2,
            0,2,3,
        ];

        this.flaps = [];

        var ailGeometry = new THREE.BufferGeometry();
        ailGeometry.setIndex( ailIndices );
        ailGeometry.setAttribute( 'position', new THREE.BufferAttribute( ailVertices, 3 ) );
        var darkGreenMat = new THREE.MeshBasicMaterial({ color: 0x006400, side: THREE.DoubleSide });

        var elevGeometry = new THREE.BufferGeometry();
        elevGeometry.setIndex( ailIndices );
        elevGeometry.setAttribute( 'position', new THREE.BufferAttribute( ailVertices.map(value => value*0.5), 3 ) );
        var darkBlueMat = new THREE.MeshBasicMaterial({ color: 0x000064, side: THREE.DoubleSide });

        var rudGeometry = new THREE.BufferGeometry();
        rudGeometry.setIndex( rudIndices );
        rudGeometry.setAttribute( 'position', new THREE.BufferAttribute( rudVertices, 3 ) );
        var darkPurpleMat = new THREE.MeshBasicMaterial({ color: 0x663a82, side: THREE.DoubleSide });
        var rud = new THREE.Mesh(rudGeometry, darkPurpleMat);
        rud.position.x = -length*1.85;

        this.obj.add(rud);

        for (let i = 0; i < 2; i++) {
            var ailMesh = new THREE.Mesh(ailGeometry, darkGreenMat);
            var elevMesh = new THREE.Mesh(elevGeometry, darkBlueMat);
            ailMesh.position.x = ailLength;
            elevMesh.position.x = -2*length + ailLength*0.5;
            switch (i) {
                case 0:
                    ailMesh.position.y = -width/2;
                    elevMesh.position.y = -width/4;
                    break;
                case 1:
                    ailMesh.position.y = width/2 - ailWidth;
                    elevMesh.position.y = width/4 - ailWidth/2;
                    break;
            }
            this.flaps.push(ailMesh);
            this.flaps.push(elevMesh);
            this.obj.add(ailMesh);
            this.obj.add(elevMesh);
        }
        this.flaps.push(rud);

        var rotorLines = [];
        this.thrustArrows = [];
        for (let i = 0; i < ys.length; i++) {
            var line = new THREE.LineSegments(edges, mat);
            this.obj.add(line);
            rotorLines.push(line);
            rotorLines[i].position.y = ys[i];
            rotorLines[i].position.x = xs[i];
            rotorLines[i].rotation.y = 3.1415/2;
            var arrow = new THREE.ArrowHelper(
                new THREE.Vector3( 1., 0., 0. ),
                new THREE.Vector3( xs[i], ys[i], 0 ),
                0.,
                0xaa5500,
                );
            this.obj.add(arrow);
            this.thrustArrows.push(arrow);
        }
    }
    addToScene(theScene) {
        theScene.add(this.obj);
    }
    setPose(pos, quat) {
        this.obj.position.x = pos[0];
        this.obj.position.y = pos[1];
        this.obj.position.z = pos[2];
        this.obj.quaternion.w = quat[0];
        this.obj.quaternion.x = quat[1];
        this.obj.quaternion.y = quat[2];
        this.obj.quaternion.z = quat[3];
    }
    setControls(controls) {
        for (let i = 0; i < this.nRotors; i++) {
            if (i >= controls.length)
                break;

            let arrowLength = 2.*controls[i]*this.diameter
            let arrowHeadLength = arrowLength < 0.4*this.diameter ? arrowLength : 0.4*this.diameter;
            this.thrustArrows[i].setLength(
                arrowLength,
                arrowHeadLength,
                0.4*this.diameter,
                );
        }
        for (let i = 0; i < this.nFlaps; i++) {
            var idx = i + this.nRotors;
            if (idx >= controls.length)
                break;

            switch (i) {
                case 4:
                    this.flaps[i].rotation.z = controls[idx]*0.8;
                    break;
                default:
                    this.flaps[i].rotation.y = controls[idx]*0.8;
                    break;
            }
        }
    }
}

