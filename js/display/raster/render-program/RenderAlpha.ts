// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for alpha (an opacity) applied to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderPath, RenderPathProgram, RenderProgram, SerializedRenderProgram, constantTrue, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { SerializedRenderPath } from './RenderPath.js';

export default class RenderAlpha extends RenderPathProgram {
  public constructor(
    path: RenderPath | null,
    public readonly program: RenderProgram,
    public readonly alpha: number
  ) {
    super( path );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderAlpha( this.getTransformedPath( transform ), this.program.transformed( transform ), this.alpha );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
           other instanceof RenderAlpha &&
           this.program.equals( other.program ) &&
           this.alpha === other.alpha;
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderAlpha( this.path, this.program.replace( callback ), this.alpha );
    }
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.program.depthFirst( callback );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    if ( this.path ) {
      return this.program.isFullyTransparent();
    }
    else {
      return this.alpha === 0 || this.program.isFullyTransparent();
    }
  }

  public override isFullyOpaque(): boolean {
    return this.alpha === 1 && this.program.isFullyOpaque();
  }

  public override needsFace(): boolean {
    return this.program.needsFace();
  }

  public override needsArea(): boolean {
    return this.program.needsArea();
  }

  public override needsCentroid(): boolean {
    return this.program.needsCentroid();
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );
    if ( program.isFullyTransparent() || this.alpha === 0 ) {
      return RenderColor.TRANSPARENT;
    }

    // No difference inside-outside
    if ( this.alpha === 1 || !this.isInPath( pathTest ) ) {
      return program;
    }

    // Now we're "inside" our path
    if ( program instanceof RenderColor ) {
      return new RenderColor( null, program.color.timesScalar( this.alpha ) );
    }
    else {
      return new RenderAlpha( null, program, this.alpha );
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
    const source = this.program.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );

    if ( this.isInPath( pathTest ) ) {
      return source.timesScalar( this.alpha );
    }
    else {
      return source;
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderAlpha (${this.path ? this.path.id : 'null'}, alpha:${this.alpha})\n` +
           `${this.program.toRecursiveString( indent + '  ' )}`;
  }

  public override serialize(): SerializedRenderAlpha {
    return {
      type: 'RenderAlpha',
      path: this.path ? this.path.serialize() : null,
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderAlpha ): RenderAlpha {
    return new RenderAlpha( obj.path ? RenderPath.deserialize( obj.path ) : null, RenderProgram.deserialize( obj.program ), 1 );
  }
}

scenery.register( 'RenderAlpha', RenderAlpha );

export type SerializedRenderAlpha = {
  type: 'RenderAlpha';
  path: SerializedRenderPath | null;
  program: SerializedRenderProgram;
};
