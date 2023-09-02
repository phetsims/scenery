// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a triangular barycentric blend. Applies perspective correction.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Vector3 from '../../../../../dot/js/Vector3.js';

export enum RenderBarycentricPerspectiveBlendAccuracy {
  // TODO: Accurate should really be the version that runs the perspective correction integral!!!!
  // TODO: do this, the math should work!
  Centroid = 0,
  PixelCenter = 1
}

scenery.register( 'RenderBarycentricPerspectiveBlendAccuracy', RenderBarycentricPerspectiveBlendAccuracy );

export default class RenderBarycentricPerspectiveBlend extends RenderProgram {

  public constructor(
    public readonly pointA: Vector3,
    public readonly pointB: Vector3,
    public readonly pointC: Vector3,
    public readonly accuracy: RenderBarycentricPerspectiveBlendAccuracy,
    public readonly a: RenderProgram,
    public readonly b: RenderProgram,
    public readonly c: RenderProgram
  ) {
    assert && assert( pointA.isFinite() );
    assert && assert( pointB.isFinite() );
    assert && assert( pointC.isFinite() );
    assert && assert( !pointA.toVector2().equals( pointB.toVector2() ) );
    assert && assert( !pointB.toVector2().equals( pointC.toVector2() ) );
    assert && assert( !pointC.toVector2().equals( pointA.toVector2() ) );
    assert && assert( pointA.z > 0 && pointB.z > 0 && pointC.z > 0, 'All points must be in front of the camera' );

    super();
  }

  public override getName(): string {
    return 'RenderBarycentricPerspectiveBlend';
  }

  public override getChildren(): RenderProgram[] {
    return [ this.a, this.b, this.c ];
  }

  public override withChildren( children: RenderProgram[] ): RenderBarycentricPerspectiveBlend {
    assert && assert( children.length === 3 );
    return new RenderBarycentricPerspectiveBlend( this.pointA, this.pointB, this.pointC, this.accuracy, children[ 0 ], children[ 1 ], children[ 2 ] );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    const xyA = transform.timesVector2( this.pointA.toVector2() );
    const xyB = transform.timesVector2( this.pointB.toVector2() );
    const xyC = transform.timesVector2( this.pointC.toVector2() );

    return new RenderBarycentricPerspectiveBlend(
      new Vector3( xyA.x, xyA.y, this.pointA.z ),
      new Vector3( xyB.x, xyB.y, this.pointB.z ),
      new Vector3( xyC.x, xyC.y, this.pointC.z ),
      this.accuracy,
      this.a.transformed( transform ),
      this.b.transformed( transform ),
      this.c.transformed( transform )
    );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.pointA.equals( other.pointA ) &&
           this.pointB.equals( other.pointB ) &&
           this.pointC.equals( other.pointC ) &&
           this.accuracy === other.accuracy;
  }

  public override needsCentroid(): boolean {
    return this.accuracy === RenderBarycentricPerspectiveBlendAccuracy.Centroid || super.needsCentroid();
  }

  public override simplified(): RenderProgram {
    const a = this.a.simplified();
    const b = this.b.simplified();
    const c = this.c.simplified();

    if ( a.isFullyTransparent() && b.isFullyTransparent() && c.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    if ( a.equals( b ) && a.equals( c ) ) {
      return a;
    }

    if ( this.a !== a || this.b !== b || this.c !== c ) {
      return new RenderBarycentricPerspectiveBlend( this.pointA, this.pointB, this.pointC, this.accuracy, a, b, c );
    }
    else {
      return this;
    }
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

    const point = this.accuracy === RenderBarycentricPerspectiveBlendAccuracy.Centroid ? centroid : new Vector2( ( minX + maxX ) / 2, ( minY + maxY ) / 2 );
    const pA = this.pointA;
    const pB = this.pointB;
    const pC = this.pointC;

    // TODO: can precompute things like this!!!
    // TODO: factor out common things!
    const det = ( pB.y - pC.y ) * ( pA.x - pC.x ) + ( pC.x - pB.x ) * ( pA.y - pC.y );

    const lambdaA = ( ( pB.y - pC.y ) * ( point.x - pC.x ) + ( pC.x - pB.x ) * ( point.y - pC.y ) ) / det;
    const lambdaB = ( ( pC.y - pA.y ) * ( point.x - pC.x ) + ( pA.x - pC.x ) * ( point.y - pC.y ) ) / det;
    const lambdaC = 1 - lambdaA - lambdaB;

    const aColor = this.a.evaluate( face, area, centroid, minX, minY, maxX, maxY ).timesScalar( 1 / pA.z );
    const bColor = this.b.evaluate( face, area, centroid, minX, minY, maxX, maxY ).timesScalar( 1 / pB.z );
    const cColor = this.c.evaluate( face, area, centroid, minX, minY, maxX, maxY ).timesScalar( 1 / pC.z );
    const z = 1 / ( lambdaA / pA.z + lambdaB / pB.z + lambdaC / pC.z );

    assert && assert( aColor.isFinite() && bColor.isFinite() && cColor.isFinite() && z > 0, 'All colors must be finite and z must be positive' );
    assert && assert( isFinite( lambdaA ) && isFinite( lambdaB ) && isFinite( lambdaC ), 'Lambdas must be finite' );

    return new Vector4(
      aColor.x * lambdaA + bColor.x * lambdaB + cColor.x * lambdaC,
      aColor.y * lambdaA + bColor.y * lambdaB + cColor.y * lambdaC,
      aColor.z * lambdaA + bColor.z * lambdaB + cColor.z * lambdaC,
      aColor.w * lambdaA + bColor.w * lambdaB + cColor.w * lambdaC
    ).timesScalar( z );
  }

  public override serialize(): SerializedRenderBarycentricPerspectiveBlend {
    return {
      type: 'RenderBarycentricPerspectiveBlend',
      pointA: [ this.pointA.x, this.pointA.y, this.pointA.z ],
      pointB: [ this.pointB.x, this.pointB.y, this.pointB.z ],
      pointC: [ this.pointC.x, this.pointC.y, this.pointC.z ],
      accuracy: this.accuracy,
      a: this.a.serialize(),
      b: this.b.serialize(),
      c: this.c.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderBarycentricPerspectiveBlend ): RenderBarycentricPerspectiveBlend {
    return new RenderBarycentricPerspectiveBlend(
      new Vector3( obj.pointA[ 0 ], obj.pointA[ 1 ], obj.pointA[ 2 ] ),
      new Vector3( obj.pointB[ 0 ], obj.pointB[ 1 ], obj.pointB[ 2 ] ),
      new Vector3( obj.pointC[ 0 ], obj.pointC[ 1 ], obj.pointC[ 2 ] ),
      obj.accuracy,
      RenderProgram.deserialize( obj.a ),
      RenderProgram.deserialize( obj.b ),
      RenderProgram.deserialize( obj.c )
    );
  }
}

scenery.register( 'RenderBarycentricPerspectiveBlend', RenderBarycentricPerspectiveBlend );

export type SerializedRenderBarycentricPerspectiveBlend = {
  type: 'RenderBarycentricPerspectiveBlend';
  pointA: number[];
  pointB: number[];
  pointC: number[];
  accuracy: RenderBarycentricPerspectiveBlendAccuracy;
  a: SerializedRenderProgram;
  b: SerializedRenderProgram;
  c: SerializedRenderProgram;
};
