// Copyright 2023, University of Colorado Boulder

/**
 * Represents an abstract rendering program, that may be location-varying
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderPath, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default abstract class RenderProgram {
  public abstract isFullyTransparent(): boolean;

  public abstract isFullyOpaque(): boolean;

  public abstract transformed( transform: Matrix3 ): RenderProgram;

  public abstract simplify( pathTest?: ( renderPath: RenderPath ) => boolean ): RenderProgram;

  // Premultiplied linear RGB, ignoring the path
  public abstract evaluate(
    face: ClippableFace | null, // if null, it is fully covered
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    pathTest?: ( renderPath: RenderPath ) => boolean
  ): Vector4;

  public abstract toRecursiveString( indent: string ): string;

  public abstract equals( other: RenderProgram ): boolean;

  public abstract replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram;

  public depthFirst( callback: ( program: RenderProgram ) => void ): void {
    callback( this );
  }
}
scenery.register( 'RenderProgram', RenderProgram );
