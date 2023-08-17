// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a linear blend (essentially a chunk of a linear gradient with only a linear transition between
 * two things.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { constantTrue, RenderColor, RenderColorSpace, RenderPath, RenderPathProgram, RenderProgram, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

const scratchLinearBlendVector = new Vector2( 0, 0 );

export default class RenderLinearBlend extends RenderPathProgram {

  public constructor(
    path: RenderPath | null,
    public readonly scaledNormal: Vector2,
    public readonly offset: number,
    public readonly zero: RenderProgram,
    public readonly one: RenderProgram,
    public readonly colorSpace: RenderColorSpace
  ) {
    assert && assert( scaledNormal.isFinite() && scaledNormal.magnitude > 0 );
    assert && assert( isFinite( offset ) );

    super( path );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    // scaledNormal dot startPoint = offset
    // scaledNormal dot endPoint = offset + 1

    // scaledNormal dot ( offset * inverseScaledNormal ) = offset
    // scaledNormal dot ( ( offset + 1 ) * inverseScaledNormal ) = offset + 1

    const beforeStartPoint = this.scaledNormal.timesScalar( this.offset / this.scaledNormal.magnitudeSquared );
    const beforeEndPoint = this.scaledNormal.timesScalar( ( this.offset + 1 ) / this.scaledNormal.magnitudeSquared );

    const afterStartPoint = transform.timesVector2( beforeStartPoint );
    const afterEndPoint = transform.timesVector2( beforeEndPoint );
    const afterDelta = afterEndPoint.minus( afterStartPoint );

    const afterNormal = afterDelta.normalized().timesScalar( 1 / afterDelta.magnitude );
    const afterOffset = afterNormal.dot( afterStartPoint );

    assert && assert( Math.abs( afterNormal.dot( afterEndPoint ) - afterOffset - 1 ) < 1e-8, 'afterNormal.dot( afterEndPoint ) - afterOffset' );

    return new RenderLinearBlend(
      this.getTransformedPath( transform ),
      afterNormal,
      afterOffset,
      this.zero.transformed( transform ),
      this.one.transformed( transform ),
      this.colorSpace
    );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
      other instanceof RenderLinearBlend &&
      this.scaledNormal.equals( other.scaledNormal ) &&
      this.offset === other.offset &&
      this.zero.equals( other.zero ) &&
      this.one.equals( other.one ) &&
      this.colorSpace === other.colorSpace;
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderLinearBlend( this.path, this.scaledNormal, this.offset, this.zero.replace( callback ), this.one.replace( callback ), this.colorSpace );
    }
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.zero.depthFirst( callback );
    this.one.depthFirst( callback );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    return this.zero.isFullyTransparent() && this.one.isFullyTransparent();
  }

  public override isFullyOpaque(): boolean {
    return this.path === null && this.zero.isFullyOpaque() && this.one.isFullyOpaque();
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const zero = this.zero.simplify( pathTest );
    const one = this.one.simplify( pathTest );

    if ( zero.isFullyTransparent() && one.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    if ( this.isInPath( pathTest ) ) {
      return new RenderLinearBlend( null, this.scaledNormal, this.offset, zero, one, this.colorSpace );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): Vector4 {
    if ( !this.isInPath( pathTest ) ) {
      return Vector4.ZERO;
    }

    const localPoint = scratchLinearBlendVector.set( point );

    const t = this.scaledNormal.dot( localPoint ) - this.offset;

    if ( t <= 0 ) {
      return this.zero.evaluate( point, pathTest );
    }
    else if ( t >= 1 ) {
      return this.one.evaluate( point, pathTest );
    }
    else {
      return RenderColor.ratioBlend(
        this.zero.evaluate( point, pathTest ),
        this.one.evaluate( point, pathTest ),
        t,
        this.colorSpace
      );
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderLinearBlend (${this.path ? this.path.id : 'null'})`;
  }
}

scenery.register( 'RenderLinearBlend', RenderLinearBlend );
