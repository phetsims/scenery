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
  var Image = require( 'SCENERY/nodes/Image' );

  var dirty = false;
  /**
   * Convert a single scenery node to a pixi node (without the children, which are handled in toPixi)
   */
  var toPixiWithoutChildren = function( sceneryNode ) {
    var i;
    if ( sceneryNode instanceof Path ) {

      var segment;

      // TODO: Use Pixi.Shape where possible
      var path = sceneryNode;
      var graphics = new PIXI.Graphics();

      var shape = path.shape;
      if ( shape !== null ) {
        graphics.beginFill( path.getFillColor().toNumber() );
        for ( i = 0; i < shape.subpaths.length; i++ ) {
          var subpath = shape.subpaths[ i ];
          for ( var k = 0; k < subpath.segments.length; k++ ) {
            segment = subpath.segments[ k ];
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
      }

      return graphics;
    }
    else if ( sceneryNode instanceof Text ) {
      return new PIXI.Text( sceneryNode.text );
    }
    else if ( sceneryNode instanceof Image ) {
      var texture = PIXI.Texture.fromCanvas( sceneryNode.image );
      return new PIXI.Sprite( texture );
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

    var listener = function() {
      pixiNode.position.x = sceneryNode.x;
      pixiNode.position.y = sceneryNode.y;
      dirty = true;
    };
    sceneryNode.on( 'transform', listener );

    // Get the correct initial values
    listener();

    for ( var i = 0; i < sceneryNode._children.length; i++ ) {
      pixiNode.addChild( toPixi( sceneryNode._children[ i ] ) );
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

    this.sceneryRootNode = sceneryRootNode;

    // Create the Pixi Stage
    this.stage = new PIXI.Stage( 0xFFFFFF );

    // Convert the scenery node to Pixi and add it to the stage
    this.stage.addChild( toPixi( sceneryRootNode ) );

    // Create the renderer and view
    this.pixiRenderer = PIXI.autoDetectRenderer( 1024, 768, { transparent: true } );

    // Initial draw
    this.pixiRenderer.render( this.stage );

    // Show the canvas in the DOM
    DOM.call( this, this.pixiRenderer.view, options );
  };

  return inherit( DOM, scenery.PixiNode, {
    render: function() {
      if ( dirty ) {
        this.pixiRenderer.render( this.stage );
        dirty = false;
      }
    },

    // TODO: deltas not recreate-the-world-each-frame
    sync: function() {
      this.stage.removeChild( this.stage.children[ 0 ] );
      var pixiNode = toPixi( this.sceneryRootNode );
      //this.stage.addChild( pixiNode.children[0] );
      this.stage.addChild( pixiNode );
      this.dirty = true;
      this.render();
    }
  } );
} );