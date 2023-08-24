// Copyright 2023, University of Colorado Boulder

/**
 * Convert a Node to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Color, ColorMatrixFilter, CombinedRaster, Display, Image, LinearGradient, Node, Path, Pattern, RadialGradient, Rasterize, RenderAlpha, RenderBlendCompose, RenderColor, RenderFilter, RenderGradientStop, RenderImage, RenderImageable, RenderLinearGradient, RenderLinearGradientAccuracy, RenderPath, RenderProgram, RenderRadialGradient, RenderRadialGradientAccuracy, scenery, Sprites, TColor, Text, TPaint } from '../../../imports.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import RenderComposeType from './RenderComposeType.js';
import RenderBlendType from './RenderBlendType.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { Shape } from '../../../../../kite/js/imports.js';
import { isTReadOnlyProperty } from '../../../../../axon/js/TReadOnlyProperty.js';
import RenderExtend from './RenderExtend.js';
import RenderColorSpace from './RenderColorSpace.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import ArialBoldFont from '../../vello/ArialBoldFont.js';
import ArialFont from '../../vello/ArialFont.js';
import RenderResampleType from './RenderResampleType.js';

// TODO: better for this?
const piecewiseOptions = {
  minLevels: 1,
  maxLevels: 10,
  // distanceEpsilon: 0.02,
  distanceEpsilon: 0.0002,
  curveEpsilon: 0.2
};

const resampleType = RenderResampleType.Bilinear;
// const resampleType = RenderResampleType.AnalyticBilinear;

const combine = ( a: RenderProgram, b: RenderProgram ) => new RenderBlendCompose(
  RenderComposeType.Over,
  RenderBlendType.Normal,
  a, b
);

// const boundsToRenderPath = ( bounds: Bounds2 ) => new RenderPath(
//   'nonzero',
//   [
//     [
//       new Vector2( bounds.minX, bounds.minY ),
//       new Vector2( bounds.maxX, bounds.minY ),
//       new Vector2( bounds.maxX, bounds.maxY ),
//       new Vector2( bounds.minX, bounds.maxY )
//     ]
//   ]
// );

const shapeToRenderPath = ( shape: Shape ) => new RenderPath(
  'nonzero',
  shape.subpaths.map( subpath => {
    return subpath.toPiecewiseLinear( piecewiseOptions ).segments.map( line => {
      return line.start;
    } );
  } )
);

const shapesToRenderPath = ( shapes: Shape[] ) => new RenderPath(
  'nonzero',
  shapes.flatMap( shape => shapeToRenderPath( shape ).subpaths )
);

const renderPathPaintToRenderProgram = ( renderPath: RenderPath, paint: TPaint, matrix: Matrix3 ): RenderProgram => {
  if ( isTReadOnlyProperty( paint ) ) {
    paint = paint.value;
  }

  if ( paint === null ) {
    return RenderColor.TRANSPARENT;
  }

  if ( typeof paint === 'string' ) {
    paint = new Color( paint );
  }

  if ( paint instanceof Color ) {
    return RenderColor.fromColor( renderPath, paint );
  }
  else {
    const paintMatrix = paint.transformMatrix ? matrix.timesMatrix( paint.transformMatrix ) : matrix;
    if ( paint instanceof LinearGradient ) {
      return new RenderLinearGradient(
        renderPath,
        paintMatrix,
        paint.start,
        paint.end,
        paint.stops.map( stop => {
          return new RenderGradientStop( stop.ratio, new RenderColor( null, colorFromTColor( stop.color ) ) );
        } ),
        RenderExtend.Pad,
        RenderColorSpace.SRGB,
        RenderLinearGradientAccuracy.SplitAccurate
      );
    }
    else if ( paint instanceof RadialGradient ) {
      return new RenderRadialGradient(
        renderPath,
        paintMatrix,
        paint.start,
        paint.startRadius,
        paint.end,
        paint.endRadius,
        paint.stops.map( stop => {
          return new RenderGradientStop( stop.ratio, new RenderColor( null, colorFromTColor( stop.color ) ) );
        } ),
        RenderExtend.Pad,
        RenderColorSpace.SRGB,
        RenderRadialGradientAccuracy.SplitAccurate
      );
    }
    else if ( paint instanceof Pattern ) {
      return new RenderImage(
        renderPath,
        paintMatrix,
        imagelikeToRenderImageable( paint.image ),
        RenderExtend.Repeat,
        RenderExtend.Repeat,
        resampleType
      );
    }
  }

  // If unimplemented
  console.log( 'SOME PAINT TYPE UNIMPLEMENTED?!?' );
  return RenderColor.TRANSPARENT;
};

const imagelikeToRenderImageable = ( imagelike: HTMLImageElement | HTMLCanvasElement | ImageBitmap ): RenderImageable => {
  const canvas = document.createElement( 'canvas' );
  canvas.width = imagelike.width;
  canvas.height = imagelike.height;
  const context = canvas.getContext( '2d', {
    willReadFrequently: true
  } )!;
  context.drawImage( imagelike, 0, 0 );

  const imageData = context.getImageData( 0, 0, canvas.width, canvas.height );

  let isFullyOpaque = true;

  const linearPremultipliedData: Vector4[] = [];
  for ( let i = 0; i < imageData.data.length / 4; i++ ) {
    const baseIndex = i * 4;
    const r = imageData.data[ baseIndex ] / 255;
    const g = imageData.data[ baseIndex + 1 ] / 255;
    const b = imageData.data[ baseIndex + 2 ] / 255;
    const a = imageData.data[ baseIndex + 3 ] / 255;
    if ( a < 1 ) {
      isFullyOpaque = false;
    }
    const srgb = new Vector4( r, g, b, a );
    const linear = RenderColor.sRGBToLinear( srgb );
    const premultiplied = RenderColor.premultiply( linear );
    linearPremultipliedData.push( premultiplied );
  }

  return {
    width: imageData.width,
    height: imageData.height,
    colorSpace: RenderColorSpace.LinearUnpremultipliedSRGB,
    isFullyOpaque: isFullyOpaque,
    evaluate: ( x, y ) => {
      return linearPremultipliedData[ y * imageData.width + x ];
    }
  };
};

const colorFromTColor = ( paint: TColor ): Vector4 => {
  if ( isTReadOnlyProperty( paint ) ) {
    paint = paint.value;
  }

  if ( paint === null ) {
    return Vector4.ZERO;
  }

  if ( typeof paint === 'string' ) {
    paint = new Color( paint );
  }

  return RenderColor.colorToPremultipliedLinear( paint );
};

export default class RenderFromNode {
  public static nodeToRenderProgram( node: Node, matrix: Matrix3 = Matrix3.IDENTITY ): RenderProgram {
    let result: RenderProgram = RenderColor.TRANSPARENT;

    const addResult = ( renderProgram: RenderProgram ) => {
      if ( !renderProgram.isFullyTransparent() ) {
        if ( result.isFullyTransparent() ) {
          result = renderProgram;
        }
        else {
          result = combine( renderProgram, result );
        }
      }
    };

    if ( !node.visible ) {
      return result;
    }

    if ( node.matrix ) {
      matrix = matrix.timesMatrix( node.matrix );
    }

    if ( node instanceof Path ) {
      const addShape = ( shape: Shape, paint: TPaint ) => {
        const renderPath = shapeToRenderPath( shape.transformed( matrix ) );

        addResult( renderPathPaintToRenderProgram( renderPath, paint, matrix ) );
      };

      if ( node.hasFill() ) {
        const shape = node.getShape();
        shape && addShape( shape, node.getFill() );
      }
      if ( node.hasStroke() ) {
        addShape( node.getStrokedShape(), node.getStroke() );
      }
    }
    else if ( node instanceof Text ) {
      const font = ( node._font.weight === 'bold' ? ArialBoldFont : ArialFont );
      const scale = node._font.numericSize / font.unitsPerEM;
      const sizedMatrix = matrix.timesMatrix( Matrix3.scaling( scale ) );

      const shapedText = font.shapeText( node.renderedText, true );

      // TODO: isolate out if we're using this
      const flipMatrix = Matrix3.rowMajor(
        1, 0, 0,
        0, -1, 0, // vertical flip
        0, 0, 1
      );

      if ( shapedText ) {
        const glyphShapes: Shape[] = [];

        let x = 0;
        shapedText.forEach( glyph => {
          const glyphMatrix = sizedMatrix.timesMatrix( Matrix3.translation( x + glyph.x, glyph.y ) ).timesMatrix( flipMatrix );

          glyphShapes.push( glyph.shape.transformed( glyphMatrix ) );

          x += glyph.advance;
        } );

        if ( node.hasFill() ) {
          const renderPath = shapesToRenderPath( glyphShapes );
          addResult( renderPathPaintToRenderProgram( renderPath, node.getFill(), matrix ) );
        }

        if ( node.hasStroke() ) {
          const renderPath = shapesToRenderPath( glyphShapes.map( shape => {
            return shape.getStrokedShape( node._lineDrawingStyles );
          } ) );
          addResult( renderPathPaintToRenderProgram( renderPath, node.getStroke(), matrix ) );
        }
      }
      else {
        console.log( 'TEXT UNSHAPED', node.renderedText );
      }
    }
    else if ( node instanceof Sprites ) {
      // TODO: Sprites
      console.log( 'SPRITES UNIMPLEMENTED' );
    }
    else if ( node instanceof Image ) {

      const nodeImage = node.image;
      if ( nodeImage ) {
        const renderPath = shapeToRenderPath( Shape.bounds( node.selfBounds ).transformed( matrix ) );

        const renderImage = new RenderImage(
          renderPath,
          matrix,
          imagelikeToRenderImageable( node.image ),
          RenderExtend.Pad,
          RenderExtend.Pad,
          resampleType
        );

        addResult( node.imageOpacity === 1 ? renderImage : new RenderAlpha( null, renderImage, node.imageOpacity ) );
      }
    }

    // Children
    // TODO: try to balance binary trees?
    node.children.forEach( child => {
      addResult( RenderFromNode.nodeToRenderProgram( child, matrix ) );
    } );

    // Filters are applied before
    node.filters.forEach( filter => {
      if ( filter instanceof ColorMatrixFilter ) {
        // NOTE: Apply them no matter what, we'll rely on later simplify (because filters can take transparent to NOT)
        result = new RenderFilter( null, result, filter.getMatrix(), filter.getTranslation() );
      }
    } );

    if ( node.effectiveOpacity !== 1 && !result.isFullyTransparent() ) {
      result = new RenderAlpha( null, result, node.effectiveOpacity );
    }

    if ( node.clipArea && !result.isFullyTransparent() ) {
      result = new RenderBlendCompose( RenderComposeType.In, RenderBlendType.Normal, result, new RenderColor(
        shapeToRenderPath( node.clipArea ),
        new Vector4( 1, 1, 1, 1 )
      ) );
    }

    return result;
  }

  public static addBackgroundColor( renderProgram: RenderProgram, color: Color ): RenderProgram {
    return combine( renderProgram, RenderColor.fromColor( null, color ) );
  }

  public static showSim(): void {
    const phet = 'phet';
    const display: Display = window[ phet ].joist.display;
    const program = RenderFromNode.addBackgroundColor( RenderFromNode.nodeToRenderProgram( display.rootNode ), Color.toColor( display.backgroundColor ) );
    const sizedProgram = program.transformed( Matrix3.scaling( window.devicePixelRatio ) );
    const width = display.width * window.devicePixelRatio;
    const height = display.height * window.devicePixelRatio;
    const raster = new CombinedRaster( width, height );
    Rasterize.rasterize( sizedProgram, raster, new Bounds2( 0, 0, width, height ) );
    const canvas = Rasterize.imageDataToCanvas( raster.toImageData() );
    canvas.style.width = `${canvas.width / window.devicePixelRatio}px`;
    canvas.style.height = `${canvas.height / window.devicePixelRatio}px`;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.zIndex = '1000000';
    document.body.appendChild( canvas );
  }

  public static nodeToJSON( node: Node ): string {
    const padding = 5;
    const addBackground = true;
    const pretty = false;
    const scale = 1;

    let program = RenderFromNode.nodeToRenderProgram( node );

    program = program.transformed( Matrix3.scaling( scale ).timesMatrix( Matrix3.translation( padding - node.bounds.minX, padding - node.bounds.minY ) ) );

    if ( addBackground ) {
      program = combine( program, new RenderImage(
        null,
        Matrix3.scaling( 5 ),
        {
          width: 2,
          height: 2,
          colorSpace: RenderColorSpace.SRGB,
          isFullyOpaque: true,
          evaluate: ( x: number, y: number ) => {
            const value = ( x + y ) % 2 === 0 ? 0.9 : 0.85;
            return new Vector4( value, value, value, 1 );
          }
        },
        RenderExtend.Repeat,
        RenderExtend.Repeat,
        RenderResampleType.NearestNeighbor
      ) );
    }

    const obj = program.serialize();
    return pretty ? JSON.stringify( obj, null, 2 ) : JSON.stringify( obj );
  }
}

scenery.register( 'RenderFromNode', RenderFromNode );
