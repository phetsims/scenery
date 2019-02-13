// Copyright 2017, University of Colorado Boulder

/**
 * Unit tests for scenery. Please run once in phet brand and once in brand=phet-io to cover all functionality.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  require( 'SCENERY/accessibility/AccessibilityTests' );
  require( 'SCENERY/accessibility/AccessibleSiblingTests' );
  require( 'SCENERY/accessibility/AccessibilityInputTests' );
  require( 'SCENERY/accessibility/AccessibilityUtilTests' );
  require( 'SCENERY/accessibility/FocusTests' );
  require( 'SCENERY/nodes/AlignBoxTests' );
  require( 'SCENERY/util/ColorTests' );
  require( 'SCENERY/display/DisplayTests' );
  require( 'SCENERY/display/FuzzTests' );
  require( 'SCENERY/listeners/DragListenerTests' );
  require( 'SCENERY/listeners/FireListenerTests' );
  require( 'SCENERY/listeners/PressListenerTests' );
  require( 'SCENERY/listeners/DragListenerTests' );
  require( 'SCENERY/accessibility/KeyStateTrackerTests' );
  require( 'SCENERY/nodes/NodeTests' );
  require( 'SCENERY/nodes/ShapeTests' );
  require( 'SCENERY/util/FontTests' );
  require( 'SCENERY/util/TrailTests' );
  require( 'SCENERY/tests/PixelComparisonTests' );
  require( 'SCENERY/tests/MiscellaneousTests' );
  const scenery = require( 'SCENERY/scenery' );

  // add elements to the QUnit fixture for our Scenery-specific tests
  // TODO: is this necessary?
  const $fixture = $( '#qunit-fixture' );
  $fixture.append( $( '<div>' ).attr( 'id', 'main' ).attr( 'style', 'position: absolute; left: 0; top: 0; background-color: white; z-index: 1; width: 640px; height: 480px;' ) );
  $fixture.append( $( '<div>' ).attr( 'id', 'secondary' ).attr( 'style', 'position: absolute; left: 0; top: 0; background-color: white; z-index: 0; width: 640px; height: 480px;' ) );

  // schema should be the same as in initializeGlobals
  const sceneryLogQueryParameter = QueryStringMachine.get( 'sceneryLog', {
    type: 'array',
    elementSchema: {
      type: 'string'
    },
    defaultValue: null
  } );
  sceneryLogQueryParameter && scenery.enableLogging( sceneryLogQueryParameter );

  // Since our tests are loaded asynchronously, we must direct QUnit to begin the tests
  QUnit.start();
} );