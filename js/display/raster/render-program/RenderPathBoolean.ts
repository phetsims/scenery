// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for alpha (an opacity) applied to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderPath, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { SerializedRenderPath } from './RenderPath.js';

export default class RenderPathBoolean extends RenderProgram {
  public constructor(
    public readonly path: RenderPath,
    public readonly inside: RenderProgram,
    public readonly outside: RenderProgram
  ) {
    super();
  }

  public override getChildren(): RenderProgram[] {
    return [ this.inside, this.outside ];
  }

  public override withChildren( children: RenderProgram[] ): RenderPathBoolean {
    assert && assert( children.length === 2 );
    return new RenderPathBoolean( this.path, children[ 0 ], children[ 1 ] );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderPathBoolean( this.path.transformed( transform ), this.inside.transformed( transform ), this.outside.transformed( transform ) );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderPathBoolean &&
           this.path === other.path &&
           this.inside.equals( other.inside ) &&
           this.outside.equals( other.outside );
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderPathBoolean( this.path, this.inside.replace( callback ), this.outside.replace( callback ) );
    }
  }

  public override isFullyTransparent(): boolean {
    return this.inside.isFullyTransparent() && this.outside.isFullyTransparent();
  }

  public override isFullyOpaque(): boolean {
    return this.inside.isFullyOpaque() && this.outside.isFullyOpaque();
  }

  public override needsFace(): boolean {
    return this.inside.needsFace() || this.outside.needsFace();
  }

  public override needsArea(): boolean {
    return this.inside.needsArea() || this.outside.needsArea();
  }

  public override needsCentroid(): boolean {
    return this.inside.needsCentroid() || this.outside.needsCentroid();
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    // TODO: partial simplification! like if inside === outside, fully transparent, etc.

    if ( pathTest( this.path ) ) {
      return this.inside.simplify( pathTest );
    }
    else {
      return this.outside.simplify( pathTest );
    }
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    pathTest: ( renderPath: RenderPath ) => boolean = constantTrue
  ): Vector4 {
    if ( pathTest( this.path ) ) {
      return this.inside.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );
    }
    else {
      return this.outside.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderPathBoolean (${this.path.id}\n` +
           `${this.inside.toRecursiveString( indent + '  ' )}\n` +
           `${this.outside.toRecursiveString( indent + '  ' )}`;
  }

  public override serialize(): SerializedRenderPathBoolean {
    return {
      type: 'RenderPathBoolean',
      path: this.path.serialize(),
      inside: this.inside.serialize(),
      outside: this.outside.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderPathBoolean ): RenderPathBoolean {
    return new RenderPathBoolean( RenderPath.deserialize( obj.path ), RenderProgram.deserialize( obj.inside ), RenderProgram.deserialize( obj.outside ) );
  }

  public static fromInside( path: RenderPath, inside: RenderProgram ): RenderPathBoolean {
    return new RenderPathBoolean( path, inside, RenderColor.TRANSPARENT );
  }
}

scenery.register( 'RenderPathBoolean', RenderPathBoolean );

export type SerializedRenderPathBoolean = {
  type: 'RenderPathBoolean';
  path: SerializedRenderPath;
  inside: SerializedRenderProgram;
  outside: SerializedRenderProgram;
};
