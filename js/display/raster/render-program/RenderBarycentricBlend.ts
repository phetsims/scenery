// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a triangular barycentric blend.
 *
 * NOTE: Does not apply perspective correction, is purely a 2d blend.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export enum RenderBarycentricBlendAccuracy {
  Accurate = 0,
  PixelCenter = 1
}

scenery.register( 'RenderBarycentricBlendAccuracy', RenderBarycentricBlendAccuracy );

export default class RenderBarycentricBlend extends RenderProgram {

  public constructor(
    public readonly pointA: Vector2,
    public readonly pointB: Vector2,
    public readonly pointC: Vector2,
    public readonly accuracy: RenderBarycentricBlendAccuracy,
    public readonly a: RenderProgram,
    public readonly b: RenderProgram,
    public readonly c: RenderProgram
  ) {
    assert && assert( pointA.isFinite() );
    assert && assert( pointB.isFinite() );
    assert && assert( pointC.isFinite() );
    assert && assert( !pointA.equals( pointB ) );
    assert && assert( !pointB.equals( pointC ) );
    assert && assert( !pointC.equals( pointA ) );

    super(
      [ a, b, c ],
      a.isFullyTransparent && b.isFullyTransparent && c.isFullyTransparent,
      a.isFullyOpaque && b.isFullyOpaque && c.isFullyOpaque,
      false,
      false,
      accuracy === RenderBarycentricBlendAccuracy.Accurate
    );
  }

  public override getName(): string {
    return 'RenderBarycentricBlend';
  }

  public override withChildren( children: RenderProgram[] ): RenderBarycentricBlend {
    assert && assert( children.length === 3 );
    return new RenderBarycentricBlend( this.pointA, this.pointB, this.pointC, this.accuracy, children[ 0 ], children[ 1 ], children[ 2 ] );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderBarycentricBlend(
      transform.timesVector2( this.pointA ),
      transform.timesVector2( this.pointB ),
      transform.timesVector2( this.pointC ),
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

  public override simplified(): RenderProgram {
    const a = this.a.simplified();
    const b = this.b.simplified();
    const c = this.c.simplified();

    if ( a.isFullyTransparent && b.isFullyTransparent && c.isFullyTransparent ) {
      return RenderColor.TRANSPARENT;
    }

    if ( a.equals( b ) && a.equals( c ) ) {
      return a;
    }

    if ( this.a !== a || this.b !== b || this.c !== c ) {
      return new RenderBarycentricBlend( this.pointA, this.pointB, this.pointC, this.accuracy, a, b, c );
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

    const point = this.accuracy === RenderBarycentricBlendAccuracy.Accurate ? centroid : new Vector2( ( minX + maxX ) / 2, ( minY + maxY ) / 2 );
    const pA = this.pointA;
    const pB = this.pointB;
    const pC = this.pointC;

    // TODO: can precompute things like this!!!
    // TODO: factor out common things!
    const det = ( pB.y - pC.y ) * ( pA.x - pC.x ) + ( pC.x - pB.x ) * ( pA.y - pC.y );

    const lambdaA = ( ( pB.y - pC.y ) * ( point.x - pC.x ) + ( pC.x - pB.x ) * ( point.y - pC.y ) ) / det;
    const lambdaB = ( ( pC.y - pA.y ) * ( point.x - pC.x ) + ( pA.x - pC.x ) * ( point.y - pC.y ) ) / det;
    const lambdaC = 1 - lambdaA - lambdaB;

    const aColor = this.a.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    const bColor = this.b.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    const cColor = this.c.evaluate( face, area, centroid, minX, minY, maxX, maxY );

    return new Vector4(
      aColor.x * lambdaA + bColor.x * lambdaB + cColor.x * lambdaC,
      aColor.y * lambdaA + bColor.y * lambdaB + cColor.y * lambdaC,
      aColor.z * lambdaA + bColor.z * lambdaB + cColor.z * lambdaC,
      aColor.w * lambdaA + bColor.w * lambdaB + cColor.w * lambdaC
    );
  }

  public override serialize(): SerializedRenderBarycentricBlend {
    return {
      type: 'RenderBarycentricBlend',
      pointA: [ this.pointA.x, this.pointA.y ],
      pointB: [ this.pointB.x, this.pointB.y ],
      pointC: [ this.pointC.x, this.pointC.y ],
      accuracy: this.accuracy,
      a: this.a.serialize(),
      b: this.b.serialize(),
      c: this.c.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderBarycentricBlend ): RenderBarycentricBlend {
    return new RenderBarycentricBlend(
      new Vector2( obj.pointA[ 0 ], obj.pointA[ 1 ] ),
      new Vector2( obj.pointB[ 0 ], obj.pointB[ 1 ] ),
      new Vector2( obj.pointC[ 0 ], obj.pointC[ 1 ] ),
      obj.accuracy,
      RenderProgram.deserialize( obj.a ),
      RenderProgram.deserialize( obj.b ),
      RenderProgram.deserialize( obj.c )
    );
  }
}

scenery.register( 'RenderBarycentricBlend', RenderBarycentricBlend );

export type SerializedRenderBarycentricBlend = {
  type: 'RenderBarycentricBlend';
  pointA: number[];
  pointB: number[];
  pointC: number[];
  accuracy: RenderBarycentricBlendAccuracy;
  a: SerializedRenderProgram;
  b: SerializedRenderProgram;
  c: SerializedRenderProgram;
};
