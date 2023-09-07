// Copyright 2023, University of Colorado Boulder

/**
 * Represents mesh data that can be handled in different ways.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';
import Vector3 from '../../../../../dot/js/Vector3.js';

export default class Mesh {

  public readonly vertices: Vector3[] = [];
  public readonly uvs: Vector3[] = [];
  public readonly normals: Vector3[] = [];
  public readonly faces: MeshFace[] = [];
  public readonly lines: MeshLine[] = [];

  public constructor( public readonly name: string | null = null ) {}

  public static loadOBJ( str: string ): Mesh[] {
    const lines = str.split( '\n' );

    const meshes: Mesh[] = [];
    let currentMaterial = '';
    let currentGroup = '';
    let currentSmoothingGroup = 0;

    const getCurrentMesh = () => {
      if ( meshes.length === 0 ) {
        meshes.push( new Mesh() );
      }
      return meshes[ meshes.length - 1 ];
    };

    const getVector3 = ( bits: string[] ): Vector3 => {
      return new Vector3(
        bits.length >= 2 ? parseFloat( bits[ 1 ] ) : 0.0,
        bits.length >= 3 ? parseFloat( bits[ 2 ] ) : 0.0,
        bits.length >= 4 ? parseFloat( bits[ 3 ] ) : 0.0
      );
    };

    for ( let i = 0; i < lines.length; i++ ) {
      let line = lines[ i ];
      if ( line.includes( '#' ) ) {
        line = line.substring( 0, line.indexOf( '#' ) );
      }
      line = line.replace( /\s+/g, ' ' ).trim();

      const bits = line.split( ' ' );
      if ( !bits.length ) {
        continue;
      }

      switch( bits[ 0 ].toLowerCase() ) {
        case 'o': {
          const mesh = new Mesh( bits[ 1 ] || '' );
          meshes.push( mesh );
          currentGroup = '';
          currentSmoothingGroup = 0;
          break;
        }
        case 'g':
          currentGroup = bits[ 1 ];
          break;
        case 'v':
          getCurrentMesh().vertices.push( getVector3( bits ) );
          break;
        case 'vt':
          getCurrentMesh().uvs.push( getVector3( bits ) );
          break;
        case 'vn':
          getCurrentMesh().normals.push( getVector3( bits ) );
          break;
        case 'l': {
          const vertexCount = bits.length - 1;

          const line = new MeshLine();

          for ( let i = 0; i < vertexCount; i++ ) {
            const vertexBits = bits[ i + 1 ].split( '/' );

            let vertexIndex = parseInt( vertexBits[ 0 ], 10 );
            if ( vertexIndex < 0 ) {
              vertexIndex = getCurrentMesh().vertices.length + vertexIndex + 1;
            }
            line.vertexIndices.push( vertexIndex - 1 );

            line.uvIndices.push(
              vertexBits.length > 1 && vertexBits[ 1 ] ? parseInt( vertexBits[ 1 ], 10 ) - 1 : vertexIndex - 1
            );
          }
          getCurrentMesh().lines.push( line );
          break;
        }
        case 's':
          currentSmoothingGroup = ( bits[ 1 ].toLowerCase() === 'off' ) ? 0 : parseInt( bits[ 1 ], 10 );
          break;
        case 'f': {
          const vertexCount = bits.length - 1;

          const face = new MeshFace( currentMaterial, currentGroup, currentSmoothingGroup );

          for ( let i = 0; i < vertexCount; i++ ) {
            const vertexBits = bits[ i + 1 ].split( '/' );

            let vertexIndex = parseInt( vertexBits[ 0 ], 10 );
            if ( vertexIndex < 0 ) {
              vertexIndex = getCurrentMesh().vertices.length + vertexIndex + 1;
            }
            face.vertexIndices.push( vertexIndex - 1 );

            face.uvIndices.push(
              vertexBits.length > 1 && vertexBits[ 1 ] ? parseInt( vertexBits[ 1 ], 10 ) - 1 : vertexIndex - 1
            );
            face.normalIndices.push(
              vertexBits.length > 2 ? parseInt( vertexBits[ 2 ], 10 ) - 1 : vertexIndex - 1
            );
          }

          getCurrentMesh().faces.push( face );
          break;
        }
        case 'mtllib': // Reference to a material library file (.mtl)
          break;
        case 'usemtl': // Sets the current material to be applied to polygons defined from this point forward
          if ( bits.length >= 2 ) {
            currentMaterial = bits[ 1 ];
          }
          break;
        default:
      }
    }

    return meshes;
  }
}

export class MeshFace {
  public readonly vertexIndices: number[] = [];
  public readonly uvIndices: number[] = [];
  public readonly normalIndices: number[] = [];

  public constructor(
    public readonly material: string,
    public readonly group: string,
    public readonly smoothingGroup: number
  ) {}
}

export class MeshLine {
  public readonly vertexIndices: number[] = [];
  public readonly uvIndices: number[] = [];
}

scenery.register( 'Mesh', Mesh );
