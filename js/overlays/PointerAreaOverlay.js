// Copyright 2013-2015, University of Colorado Boulder

/**
 * Displays mouse and touch areas when they are customized. Expensive to display!
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

  function PointerAreaOverlay( display, rootNode ) {
    ShapeBasedOverlay.call( this, display, rootNode, 'mouseTouchAreaOverlay' );
  }

  scenery.register( 'PointerAreaOverlay', PointerAreaOverlay );

  inherit( ShapeBasedOverlay, PointerAreaOverlay, {
    // @override
    addShapes: function() {
      var self = this;

      new scenery.Trail( this.rootNode ).eachTrailUnder( function( trail ) {
        var node = trail.lastNode();
        if ( !node.isVisible() ) {
          // skip this subtree if the node is invisible
          return true;
        }
        if ( ( node.mouseArea || node.touchArea ) && trail.isVisible() ) {
          var transform = trail.getTransform();

          if ( node.mouseArea ) {
            self.addShape( transform.transformShape( node.mouseArea.isBounds ? Shape.bounds( node.mouseArea ) : node.mouseArea ), 'rgba(0,0,255,0.8)', true );
          }
          if ( node.touchArea ) {
            self.addShape( transform.transformShape( node.touchArea.isBounds ? Shape.bounds( node.touchArea ) : node.touchArea ), 'rgba(255,0,0,0.8)', false );
          }
        }
      } );
    }
  } );

  return PointerAreaOverlay;
} );
