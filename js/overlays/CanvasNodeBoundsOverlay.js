// Copyright 2013-2019, University of Colorado Boulder

/**
 * Displays CanvasNode bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const inherit = require( 'PHET_CORE/inherit' );
  const Shape = require( 'KITE/Shape' );
  const ShapeBasedOverlay = require( 'SCENERY/overlays/ShapeBasedOverlay' );

  const scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );

  function CanvasNodeBoundsOverlay( display, rootNode ) {
    ShapeBasedOverlay.call( this, display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  scenery.register( 'CanvasNodeBoundsOverlay', CanvasNodeBoundsOverlay );

  inherit( ShapeBasedOverlay, CanvasNodeBoundsOverlay, {
    // @override
    addShapes: function() {
      const self = this;

      new scenery.Trail( this.rootNode ).eachTrailUnder( function( trail ) {
        const node = trail.lastNode();
        if ( !node.isVisible() ) {
          // skip this subtree if the node is invisible
          return true;
        }
        if ( ( node instanceof scenery.CanvasNode ) && trail.isVisible() ) {
          const transform = trail.getTransform();

          self.addShape( transform.transformShape( Shape.bounds( node.selfBounds ) ), 'rgba(0,255,0,0.8)', true );
        }
      } );
    }
  } );

  return CanvasNodeBoundsOverlay;
} );
