// Copyright 2017, University of Colorado Boulder

/**
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var FocusHighlightPath = require( 'SCENERY/accessibility/FocusHighlightPath' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Shape = require( 'KITE/Shape' );

  /**
   *
   * @param {Node} node
   * @param {Object} [options]
   * @constructor
   */
  function FocusHighlightFromNode( node, options ) {

    options = _.extend( {}, options );

    this.nodeBounds = node.bounds; // TODO: bounds?

    FocusHighlightPath.call( this, null, options );

    var dilationCoefficient = this.getOuterLineWidth( node ) * 3 / 4;
    var dilatedBounds = this.nodeBounds.dilated( dilationCoefficient );

    this.setShape( Shape.bounds( dilatedBounds ) );
  }

  scenery.register( 'FocusHighlightFromNode', FocusHighlightFromNode );

  return inherit( FocusHighlightPath, FocusHighlightFromNode, {}, {




  } );
} );