//  Copyright 2002-2014, University of Colorado Boulder

/**
 * EXPERIMENTAL Pixi support using a single node that renders to a DOM node
 *
 * The changes in the scenery tree are mirrored in the pixi tree (somewhat like we have done for SVG, though with a
 * simpler pattern in this case because pixi leaves can also have children (unlike in svg)).
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var DOM = require( 'SCENERY/nodes/DOM' );
  var Path = require( 'SCENERY/nodes/Path' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   *
   * @param sceneryRootNode
   * @constructor
   */
  scenery.PixiNode = function PixiNode( sceneryRootNode, options ) {
    // iterate over the full dag and turn into pixi displayObjects
    // additionally observe all elements in the dag and update this node when they change.


    var stage = new PIXI.Stage( 0xFFFFFF );

    var renderer = PIXI.autoDetectRenderer( 400, 300, { transparent: true } );

    DOM.call( this, renderer.view, options );
    var toPixi = function( sceneryNode ) {

      // TODO: Use Pixi.Shape where possible
      if ( sceneryNode instanceof Path ) {
        var path = sceneryNode;
        var graphics = new PIXI.Graphics();

        graphics.beginFill( path.getFillColor().toNumber() );
        var shape = path.shape;

        for ( var i = 0; i < shape.subpaths.length; i++ ) {
          var subpath = shape.subpaths[ i ];
          for ( var k = 0; k < subpath.segments.length; k++ ) {
            var segment = subpath.segments[ k ];
            if ( i === 0 && k === 0 ) {
              graphics.moveTo( segment.start.x, segment.start.y );
            }
            else {
              graphics.lineTo( segment.start.x, segment.start.y );
            }

            if ( i === shape.subpaths.length - 1 && k === subpath.segments.length - 1 ) {
              graphics.lineTo( segment.end.x, segment.end.y );
            }
          }
        }

        graphics.endFill();
        return graphics;
      }
      else {
        console.log( 'unknown node type', sceneryNode );
      }
    };

    var root = toPixi( sceneryRootNode );
    stage.addChild( root );

    renderer.render( stage );

  };

  return inherit( DOM, scenery.PixiNode );
} );