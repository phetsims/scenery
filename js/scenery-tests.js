// Copyright 2017-2025, University of Colorado Boulder

/**
 * Unit tests for scenery. Please run once in phet brand and once in brand=phet-io to cover all functionality.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import qunitStart from '../../chipper/js/browser/sim-tests/qunitStart.js';
import './accessibility/FocusTests.js';
import './accessibility/KeyStateTrackerTests.js';
import './accessibility/pdom/ParallelDOMTests.js';
import './accessibility/pdom/PDOMInputTests.js';
import './accessibility/pdom/PDOMSiblingTests.js';
import './accessibility/pdom/PDOMUtilsTests.js';
import './display/DisplayTests.js';
import './display/FuzzTests.js';
import './layout/constraints/ManualConstraintTests.js';
import './layout/nodes/AlignBoxTests.js';
import './layout/nodes/FlowBoxTests.js';
import './layout/constraints/FlowConstraintTests.js';
import './layout/nodes/GridBoxTests.js';
import './layout/constraints/GridConstraintTests.js';
import './listeners/DragListenerTests.js';
import './listeners/FireListenerTests.js';
import './listeners/PressListenerTests.js';
import './listeners/KeyboardListenerTests.js';
import './input/KeyDescriptorTests.js';
import './nodes/NodeTests.js';
import './nodes/RichTextTests.js';
import './nodes/ShapeTests.js';
import './nodes/TextTests.js';
import scenery from './scenery.js';
import './tests/MiscellaneousTests.js';
import './tests/PixelComparisonTests.js';
import './util/AncestorNodesPropertyTests.js';
import './util/ColorTests.js';
import './util/DisplayedPropertyTests.js';
import './util/FontTests.js';
import './util/MatrixBetweenPropertyTests.js';
import './util/TrailTests.js';

// add elements to the QUnit fixture for our Scenery-specific tests
// TODO: is this necessary? https://github.com/phetsims/scenery/issues/1581
const fixture = document.getElementById( 'qunit-fixture' );
if ( fixture ) {
  const mainDiv = document.createElement( 'div' );
  mainDiv.id = 'main';
  mainDiv.setAttribute( 'style', 'position: absolute; left: 0; top: 0; background-color: white; z-index: 1; width: 640px; height: 480px;' );
  fixture.appendChild( mainDiv );

  const secondaryDiv = document.createElement( 'div' );
  secondaryDiv.id = 'secondary';
  secondaryDiv.setAttribute( 'style', 'position: absolute; left: 0; top: 0; background-color: white; z-index: 0; width: 640px; height: 480px;' );
  fixture.appendChild( secondaryDiv );
}

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
qunitStart();