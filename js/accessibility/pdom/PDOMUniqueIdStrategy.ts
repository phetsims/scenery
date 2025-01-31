// Copyright 2025, University of Colorado Boulder

/**
 * PDOMInstances support two different styles of unique IDs, each with their
 * own tradeoffs, https://github.com/phetsims/phet-io/issues/1851
 *
 * @author Jesse Greenberg
 */

import EnumerationValue from '../../../../phet-core/js/EnumerationValue.js';
import Enumeration from '../../../../phet-core/js/Enumeration.js';
import scenery from '../../scenery.js';

export default class PDOMUniqueIdStrategy extends EnumerationValue {
  public static readonly INDICES = new PDOMUniqueIdStrategy();
  public static readonly TRAIL_ID = new PDOMUniqueIdStrategy();

  public static readonly enumeration = new Enumeration( PDOMUniqueIdStrategy );
}
scenery.register( 'PDOMUniqueIdStrategy', PDOMUniqueIdStrategy );