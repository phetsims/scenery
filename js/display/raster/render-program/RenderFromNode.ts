// Copyright 2023, University of Colorado Boulder

/**
 * Convert a Node to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Color, ColorMatrixFilter, Display, Image, LinearGradient, Node, Path, Pattern, RadialGradient, Rasterize, RenderAlpha, RenderBlendCompose, RenderColor, RenderFilter, RenderGradientStop, RenderLinearGradient, RenderPath, RenderProgram, RenderRadialGradient, scenery, Sprites, TColor, Text, TPaint } from '../../../imports.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import RenderComposeType from './RenderComposeType.js';
import RenderBlendType from './RenderBlendType.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { Shape } from '../../../../../kite/js/imports.js';
import { isTReadOnlyProperty } from '../../../../../axon/js/TReadOnlyProperty.js';
import RenderExtend from './RenderExtend.js';
import RenderColorSpace from './RenderColorSpace.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';

// TODO: better for this?
const piecewiseOptions = {
  minLevels: 1,
  maxLevels: 10,
  // distanceEpsilon: 0.02,
  distanceEpsilon: 0.0002,
  curveEpsilon: 0.2
};

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

    // TODO: self
    if ( node instanceof Path ) {
      const addShape = ( shape: Shape, paint: TPaint ) => {

        if ( isTReadOnlyProperty( paint ) ) {
          paint = paint.value;
        }

        if ( paint === null ) {
          return;
        }

        const renderPath = shapeToRenderPath( shape.transformed( matrix ) );

        if ( typeof paint === 'string' ) {
          paint = new Color( paint );
        }

        if ( paint instanceof Color ) {
          addResult( RenderColor.fromColor( renderPath, paint ) );
        }
        else {
          const paintMatrix = paint.transformMatrix ? matrix.timesMatrix( paint.transformMatrix ) : matrix;
          if ( paint instanceof LinearGradient ) {
            addResult( new RenderLinearGradient(
              renderPath,
              paintMatrix,
              paint.start,
              paint.end,
              paint.stops.map( stop => {
                return new RenderGradientStop( stop.ratio, new RenderColor( null, colorFromTColor( stop.color ) ) );
              } ),
              RenderExtend.Pad,
              RenderColorSpace.SRGB
            ) );
          }
          else if ( paint instanceof RadialGradient ) {
            addResult( new RenderRadialGradient(
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
              RenderColorSpace.SRGB
            ) );
          }
          else if ( paint instanceof Pattern ) {
            // TODO: Patterns
            console.log( 'PATTERN UNIMPLEMENTED' );
          }
        }
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
      // TODO: Text
      console.log( 'TEXT UNIMPLEMENTED' );
    }
    else if ( node instanceof Sprites ) {
      // TODO: Sprites
      console.log( 'SPRITES UNIMPLEMENTED' );
    }
    else if ( node instanceof Image ) {
      // TODO: Image
      console.log( 'IMAGE UNIMPLEMENTED' );
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
    const debugData = Rasterize.rasterizeRenderProgram( sizedProgram, new Bounds2( 0, 0, width, height ) );
    const canvas = debugData!.canvas;
    canvas.style.width = `${canvas.width / window.devicePixelRatio}px`;
    canvas.style.height = `${canvas.height / window.devicePixelRatio}px`;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.zIndex = '1000000';
    document.body.appendChild( canvas );
  }
}

scenery.register( 'RenderFromNode', RenderFromNode );
