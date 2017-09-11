// Copyright 2013-2015, University of Colorado Boulder

/**
 * Displays CanvasNode bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Shape = require( 'KITE/Shape' );
  var ShapeBasedOverlay = require( 'SCENERY/overlays/ShapeBasedOverlay' );

  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );

  function CanvasNodeBoundsOverlay( display, rootNode ) {
    ShapeBasedOverlay.call( this, display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  scenery.register( 'CanvasNodeBoundsOverlay', CanvasNodeBoundsOverlay );

  inherit( ShapeBasedOverlay, CanvasNodeBoundsOverlay, {
    // @override
    addShapes: function() {
      var self = this;

      new scenery.Trail( this.rootNode ).eachTrailUnder( function( trail ) {
        var node = trail.lastNode();
        if ( !node.isVisible() ) {
          // skip this subtree if the node is invisible
          return true;
        }
        if ( ( node instanceof scenery.CanvasNode ) && trail.isVisible() ) {
          var transform = trail.getTransform();

          self.addShape( transform.transformShape( Shape.bounds( node.selfBounds ) ), 'rgba(0,255,0,0.8)', true );
        }
      } );
    }
  } );

  return CanvasNodeBoundsOverlay;
} );
