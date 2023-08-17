// Copyright 2023, University of Colorado Boulder

/**
 * Represents an abstract rendering program, that may be location-varying
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderPath, RenderPathProgram, scenery } from '../../../imports.js';
import { Shape } from '../../../../../kite/js/imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default abstract class RenderProgram {
  public abstract isFullyTransparent(): boolean;

  public abstract isFullyOpaque(): boolean;

  public abstract transformed( transform: Matrix3 ): RenderProgram;

  public abstract simplify( pathTest?: ( renderPath: RenderPath ) => boolean ): RenderProgram;

  // Premultiplied linear RGB, ignoring the path
  public abstract evaluate( point: Vector2, pathTest?: ( renderPath: RenderPath ) => boolean ): Vector4;

  public abstract toRecursiveString( indent: string ): string;

  public abstract equals( other: RenderProgram ): boolean;

  public abstract replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram;

  public depthFirst( callback: ( program: RenderProgram ) => void ): void {
    callback( this );
  }

  public slowRasterizeToURL( width: number, height: number, n: number ): string {
    return this.slowRasterizeToCanvas( width, height, n ).toDataURL();
  }

  public slowRasterizeToCanvas( width: number, height: number, n: number ): HTMLCanvasElement {
    const imageData = this.slowRasterizeImageData( width, height, n );
    const canvas = document.createElement( 'canvas' );
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext( '2d' )!;
    context.putImageData( imageData, 0, 0 );
    return canvas;
  }

  public slowRasterizeImageData( width: number, height: number, n: number ): ImageData {
    const imageData = new ImageData( width, height, { colorSpace: 'srgb' } );

    const paths: RenderPath[] = [];
    this.depthFirst( program => {
      if ( program instanceof RenderPathProgram && program.path !== null ) {
        paths.push( program.path );
      }
    } );

    const shapes = paths.map( path => {
      const shape = new Shape();
      path.subpaths.forEach( subpath => shape.polygon( subpath ) );
      return shape;
    } );

    for ( let y = 0; y < height; y++ ) {
      for ( let x = 0; x < width; x++ ) {
        const accumulated = new Vector4( 0, 0, 0, 0 );

        for ( let dx = 0; dx < n; dx++ ) {
          for ( let dy = 0; dy < n; dy++ ) {
            // multisample it
            const point = new Vector2( x + ( dx + 0.5 ) / n, y + ( dy + 0.5 ) / n );

            const set = new Set<RenderPath>();

            for ( let p = 0; p < paths.length; p++ ) {
              const path = paths[ p ];
              const shape = shapes[ p ];
              const inside = shape.containsPoint( point );
              if ( inside ) {
                set.add( path );
              }
            }

            const linearPremultiplied = this.evaluate( point, path => set.has( path ) );
            accumulated.add( linearPremultiplied );
          }
        }

        const scaled = accumulated.timesScalar( 1 / ( n * n ) );

        const color = RenderColor.premultipliedLinearToColor( scaled );
        imageData.data[ ( y * width + x ) * 4 ] = color.r;
        imageData.data[ ( y * width + x ) * 4 + 1 ] = color.g;
        imageData.data[ ( y * width + x ) * 4 + 2 ] = color.b;
        imageData.data[ ( y * width + x ) * 4 + 3 ] = color.a * 255;
      }
    }

    return imageData;
  }
}
scenery.register( 'RenderProgram', RenderProgram );
