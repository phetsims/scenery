// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a radial blend (essentially a chunk of a radial gradient with only a linear transition between
 * two things.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { constantTrue, RenderColor, RenderColorSpace, RenderPath, RenderPathProgram, RenderProgram, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

const scratchRadialBlendVector = new Vector2( 0, 0 );

export default class RenderRadialBlend extends RenderPathProgram {

  private readonly inverseTransform: Matrix3;

  public constructor(
    path: RenderPath | null,
    public readonly transform: Matrix3,
    public readonly radius0: number,
    public readonly radius1: number,
    public readonly zero: RenderProgram,
    public readonly one: RenderProgram,
    public readonly colorSpace: RenderColorSpace
  ) {
    assert && assert( transform.isFinite() );
    assert && assert( isFinite( radius0 ) && radius0 >= 0 );
    assert && assert( isFinite( radius1 ) && radius1 >= 0 );

    super( path );

    this.inverseTransform = transform.inverted();
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderRadialBlend(
      this.getTransformedPath( transform ),
      transform.timesMatrix( this.transform ),
      this.radius0,
      this.radius1,
      this.zero.transformed( transform ),
      this.one.transformed( transform ),
      this.colorSpace
    );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderRadialBlend &&
           this.path === other.path &&
           this.transform.equals( other.transform ) &&
           this.radius0 === other.radius0 &&
           this.radius1 === other.radius1 &&
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
      return new RenderRadialBlend( this.path, this.transform, this.radius0, this.radius1, this.zero.replace( callback ), this.one.replace( callback ), this.colorSpace );
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
      return new RenderRadialBlend( null, this.transform, this.radius0, this.radius1, zero, one, this.colorSpace );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): Vector4 {
    if ( !this.isInPath( pathTest ) ) {
      return Vector4.ZERO;
    }

    const localPoint = scratchRadialBlendVector.set( point );

    this.inverseTransform.multiplyVector2( localPoint );

    // TODO: assuming no actual order, BUT needs positive radii?
    const t = ( localPoint.magnitude - this.radius0 ) / ( this.radius1 - this.radius0 );

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
    return `${indent}RenderRadialBlend (${this.path ? this.path.id : 'null'})`;
  }
}

scenery.register( 'RenderRadialBlend', RenderRadialBlend );
