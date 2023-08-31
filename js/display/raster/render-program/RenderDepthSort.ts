// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram that provides splitting based on depth, into a RenderStack
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderPlanar, RenderProgram, SerializedRenderProgram, scenery } from '../../../imports.js';
import Matrix4 from '../../../../../dot/js/Matrix4.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Vector3 from '../../../../../dot/js/Vector3.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export default class RenderDepthSort extends RenderProgram {

  public constructor(
    public readonly items: RenderPlanar[]
  ) {
    super();
  }

  public override getName(): string {
    return 'RenderDepthSort';
  }

  public override getChildren(): RenderProgram[] {
    return this.items.map( item => item.program );
  }

  public override withChildren( children: RenderProgram[] ): RenderDepthSort {
    assert && assert( children.length === this.items.length );
    return new RenderDepthSort( children.map( ( child, i ) => {
      return new RenderPlanar( child, this.items[ i ].pointA, this.items[ i ].pointB, this.items[ i ].pointC );
    } ) );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderDepthSort( this.items.map( item => item.transformed( transform ) ) );
  }

  public override simplified(): RenderProgram {
    const items = this.items.map( item => item.withProgram( item.program.simplified() ) ).filter( item => !item.program.isFullyTransparent() );

    if ( items.length === 0 ) {
      return RenderColor.TRANSPARENT;
    }
    else if ( items.length === 1 ) {
      return items[ 0 ].program;
    }
    else if ( this.items.length !== items.length || _.some( this.items, ( item, i ) => item.program !== items[ i ].program ) ) {
      return new RenderDepthSort( items );
    }
    else {
      return this;
    }
  }

  public override isFullyOpaque(): boolean {
    return _.some( this.items, item => item.program.isFullyOpaque() );
  }

  public override isFullyTransparent(): boolean {
    return _.every( this.items, item => item.program.isFullyTransparent() );
  }

  // NOTE: If we have this (unsplit), we'll want the centroid
  public override needsCentroid(): boolean {
    return true;
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): Vector4 {

    // Negative, so that our highest-depth things are first
    const sortedItems = _.sortBy( this.items, item => -item.getDepth( centroid.x, centroid.y ) );

    const color = Vector4.ZERO.copy(); // we will mutate it

    // Blend like normal!
    for ( let i = 0; i < sortedItems.length; i++ ) {
      const blendColor = sortedItems[ i ].program.evaluate( face, area, centroid, minX, minY, maxX, maxY );
      const backgroundAlpha = 1 - blendColor.w;

      // Assume premultiplied
      color.setXYZW(
        blendColor.x + backgroundAlpha * color.x,
        blendColor.y + backgroundAlpha * color.y,
        blendColor.z + backgroundAlpha * color.z,
        blendColor.w + backgroundAlpha * color.w
      );
    }

    return color;
  }

  public override serialize(): SerializedRenderDepthSort {
    return {
      type: 'RenderDepthSort',
      items: this.items.map( item => {
        return {
          program: item.program.serialize(),
          pointA: [ item.pointA.x, item.pointA.y, item.pointA.z ],
          pointB: [ item.pointB.x, item.pointB.y, item.pointB.z ],
          pointC: [ item.pointC.x, item.pointC.y, item.pointC.z ]
        };
      } )
    };
  }

  public static override deserialize( obj: SerializedRenderDepthSort ): RenderDepthSort {
    return new RenderDepthSort( obj.items.map( item => new RenderPlanar(
      RenderProgram.deserialize( item.program ),
      new Vector3( item.pointA[ 0 ], item.pointA[ 1 ], item.pointA[ 2 ] ),
      new Vector3( item.pointB[ 0 ], item.pointB[ 1 ], item.pointB[ 2 ] ),
      new Vector3( item.pointC[ 0 ], item.pointC[ 1 ], item.pointC[ 2 ]
    ) ) ) );
  }

  public static getProjectionMatrix( near: number, far: number, minX: number, minY: number, maxX: number, maxY: number ): Matrix4 {

    const minZ = near;
    const maxZ = far;

    const diffX = maxX - minX;
    const diffY = maxY - minY;
    const diffZ = maxZ - minZ;

    return new Matrix4(
      2 * minZ / diffX, 0, -( maxX + minX ) / diffX, 0,
      0, 2 * minZ / diffY, -( maxY + minY ) / diffY, 0,
      0, 0, maxZ / diffZ, -minZ * maxZ / diffZ,
      0, 0, 1, 0
    );
  }
}

scenery.register( 'RenderDepthSort', RenderDepthSort );

export type SerializedRenderDepthSort = {
  type: 'RenderDepthSort';
  items: {
    program: SerializedRenderProgram;
    pointA: number[];
    pointB: number[];
    pointC: number[];
  }[];
};
