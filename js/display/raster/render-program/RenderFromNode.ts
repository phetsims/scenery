// Copyright 2023, University of Colorado Boulder

/**
 * Convert a Node to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Color, ColorMatrixFilter, Node, RenderAlpha, RenderBlendCompose, RenderColor, RenderFilter, RenderPath, RenderProgram, scenery } from '../../../imports.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import RenderComposeType from './RenderComposeType.js';
import RenderBlendType from './RenderBlendType.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import { Shape } from '../../../../../kite/js/imports.js';

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

const boundsToRenderPath = ( bounds: Bounds2 ) => new RenderPath(
  'nonzero',
  [
    [
      new Vector2( bounds.minX, bounds.minY ),
      new Vector2( bounds.maxX, bounds.minY ),
      new Vector2( bounds.maxX, bounds.maxY ),
      new Vector2( bounds.minX, bounds.maxY )
    ]
  ]
);

const shapeToRenderPath = ( shape: Shape ) => new RenderPath(
  'nonzero',
  shape.subpaths.map( subpath => {
    return subpath.toPiecewiseLinear( piecewiseOptions ).segments.map( line => {
      return line.start;
    } );
  } )
);

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

    // TODO: self

    if ( node.matrix ) {
      matrix = matrix.timesMatrix( node.matrix );
    }

    // Children
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
}

scenery.register( 'RenderFromNode', RenderFromNode );
