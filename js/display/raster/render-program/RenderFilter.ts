// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for applying a color-matrix filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderPath, RenderPathProgram, RenderProgram, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix4 from '../../../../../dot/js/Matrix4.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderFilter extends RenderPathProgram {
  public constructor(
    path: RenderPath | null,
    public readonly program: RenderProgram,
    public readonly matrix: Matrix4
  ) {
    super( path );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderFilter( this.getTransformedPath( transform ), this.program.transformed( transform ), this.matrix );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
           other instanceof RenderFilter &&
           this.program.equals( other.program ) &&
           this.matrix.equals( other.matrix );
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderFilter( this.path, this.program.replace( callback ), this.matrix );
    }
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.program.depthFirst( callback );
    callback( this );
  }

  // TODO: inspect matrix to see when it will maintain transparency!
  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );

    if ( this.isInPath( pathTest ) ) {
      if ( program instanceof RenderColor ) {
        return new RenderColor( null, RenderColor.premultiply( this.matrix.timesVector4( RenderColor.unpremultiply( program.color ) ) ) );
      }
      else {
        return new RenderFilter( this.path, program, this.matrix );
      }
    }
    else {
      return program;
    }
  }

  public override isFullyTransparent(): boolean {
    // TODO: color matrix check. Homogeneous?
    return false;
  }

  public override isFullyOpaque(): boolean {
    // TODO: color matrix check. Homogeneous?
    return false;
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
      return RenderColor.premultiply( this.matrix.timesVector4( RenderColor.unpremultiply( source ) ) );
    }
    else {
      return source;
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderFilter (${this.path ? this.path.id : 'null'})\n` +
           `${this.program.toRecursiveString( indent + '  ' )}`;
  }
}
scenery.register( 'RenderFilter', RenderFilter );
