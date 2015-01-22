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
  var Node = require( 'SCENERY/nodes/Node' );
  var Text = require( 'SCENERY/nodes/Text' );

  /**
   * Convert a single scenery node to a pixi node (without the children, which are handled in toPixi)
   */
  var toPixiWithoutChildren = function( sceneryNode ) {
    var i;
    if ( sceneryNode instanceof Path ) {

      // TODO: Use Pixi.Shape where possible
      var path = sceneryNode;
      var graphics = new PIXI.Graphics();

      graphics.beginFill( path.getFillColor().toNumber() );
      var shape = path.shape;

      for ( i = 0; i < shape.subpaths.length; i++ ) {
        var subpath = shape.subpaths[ i ];
        for ( var k = 0; k < subpath.segments.length; k++ ) {
          var segment = subpath.segments[ k ];
          if ( i === 0 && k === 0 ) {
            graphics.moveTo( segment.start.x, segment.start.y );
          }
          else {
            graphics.lineTo( segment.start.x, segment.start.y );
          }

          if ( k === subpath.segments.length - 1 ) {
            graphics.lineTo( segment.end.x, segment.end.y );
          }
        }

        if ( subpath.isClosed() ) {
          segment = subpath.segments[ 0 ];
          graphics.lineTo( segment.start.x, segment.start.y );
        }
      }

      graphics.endFill();

      return graphics;
    }
    else if ( sceneryNode instanceof Text ) {
      return new PIXI.Text( sceneryNode.text );
    }
    else if ( sceneryNode instanceof Node ) {

      // Handle node case last since Path, Text, Image also instanceof Node
      return new PIXI.DisplayObjectContainer();
    }
    else {
      throw new Error( 'unknown node type', sceneryNode );
    }
  };

  /**
   * Recursively convert a scenery node to a pixi DisplayObject
   * @param {Node} sceneryNode
   * @returns {PIXI.DisplayObject}
   */
  var toPixi = function toPixi( sceneryNode ) {
    var pixiNode = toPixiWithoutChildren( sceneryNode );
    for ( var i = 0; i < sceneryNode.children.length; i++ ) {
      pixiNode.addChild( toPixi( sceneryNode.children[ i ] ) );
    }
    return pixiNode;
  };

  /**
   * Iterate over the full dag and turn into pixi displayObjects
   * additionally observe all elements in the dag and update this node when they change.
   * @param sceneryRootNode
   * @param options
   * @constructor
   */
  scenery.PixiNode = function PixiNode( sceneryRootNode, options ) {

    // Create the Pixi Stage
    var stage = new PIXI.Stage( 0xFFFFFF );

    // Convert the scenery node to Pixi and add it to the stage
    stage.addChild( toPixi( sceneryRootNode ) );

    // Create the renderer and view
    var renderer = PIXI.autoDetectRenderer( 400, 300, { transparent: true } );

    // Initial draw
    renderer.render( stage );

    // Show the canvas in the DOM
    DOM.call( this, renderer.view, options );
  };

  return inherit( DOM, scenery.PixiNode );
} );