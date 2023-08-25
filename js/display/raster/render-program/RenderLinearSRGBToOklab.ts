// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert linear sRGB => Oklab
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderPath, RenderProgram, SerializedRenderProgram, constantTrue, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderLinearSRGBToOklab extends RenderProgram {
  public constructor(
    public readonly program: RenderProgram
  ) {
    super();
  }

  // TODO: How to get rid of the amounts of boilerplate here?
  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderLinearSRGBToOklab( this.program.transformed( transform ) );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderLinearSRGBToOklab &&
           this.program.equals( other.program );
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderLinearSRGBToOklab( this.program.replace( callback ) );
    }
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.program.depthFirst( callback );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    return this.program.isFullyTransparent();
  }

  public override isFullyOpaque(): boolean {
    return this.program.isFullyOpaque();
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

    if ( program.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    // Now we're "inside" our path
    if ( program instanceof RenderColor ) {
      return new RenderColor( null, RenderColor.linearToOklab( program.color ) );
    }
    else {
      return new RenderLinearSRGBToOklab( program );
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

    return RenderColor.linearToOklab( source );
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderLinearSRGBToOklab\n` +
           `${this.program.toRecursiveString( indent + '  ' )}`;
  }

  public override serialize(): SerializedRenderLinearSRGBToOklab {
    return {
      type: 'RenderLinearSRGBToOklab',
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderLinearSRGBToOklab ): RenderLinearSRGBToOklab {
    return new RenderLinearSRGBToOklab( RenderProgram.deserialize( obj.program ) );
  }
}

scenery.register( 'RenderLinearSRGBToOklab', RenderLinearSRGBToOklab );

export type SerializedRenderLinearSRGBToOklab = {
  type: 'RenderLinearSRGBToOklab';
  program: SerializedRenderProgram;
};
