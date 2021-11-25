// Copyright 2020-2021, University of Colorado Boulder

/**
 * Uses the Web Speech API to produce speech from the browser. This is a prototype, DO NOT USE IN PRODUCTION CODE.
 * There is no speech output until the voicingManager has been initialized. Supported voices will depend on platform.
 * For each voice, you can customize the rate and pitch. Only one voicingManager should be active at a time and so this
 * type is a singleton.
 *
 * @author Jesse Greenberg
 */

import BooleanProperty from '../../../../axon/js/BooleanProperty.js';
import DerivedProperty from '../../../../axon/js/DerivedProperty.js';
import Emitter from '../../../../axon/js/Emitter.js';
import EnabledComponent from '../../../../axon/js/EnabledComponent.js';
import NumberProperty from '../../../../axon/js/NumberProperty.js';
import Property from '../../../../axon/js/Property.js';
import Range from '../../../../dot/js/Range.js';
import merge from '../../../../phet-core/js/merge.js';
import stripEmbeddingMarks from '../../../../phet-core/js/stripEmbeddingMarks.js';
import Announcer from '../../../../utterance-queue/js/Announcer.js';
import Utterance from '../../../../utterance-queue/js/Utterance.js';
import { scenery, globalKeyStateTracker, KeyboardUtils } from '../../imports.js';

const DEFAULT_PRIORITY = 1;

// In ms, how frequently we will use SpeechSynthesis to keep the feature active. After long intervals without
// using SpeechSynthesis Chromebooks will take a long time to produce the next speech. Presumably it is disabling
// the feature as an optimization. But this workaround gets around it and keeps speech fast.
const ENGINE_WAKE_INTERVAL = 10000;

// In ms. In Safari, the `start` and `end` listener do not fire consistently, especially after interruption
// with cancel. But speaking behind a timeout/delay improves the behavior significantly. Timeout of 250 ms was
// determined with testing to be a good value to use. Values less than 250 broke the workaround, while larger
// values feel too sluggish. See https://github.com/phetsims/john-travoltage/issues/435
const VOICING_UTTERANCE_INTERVAL = 250;

const UTTERANCE_OPTION_DEFAULTS = {

  // {boolean} - If true and this Utterance is currently being spoken by the speech synth, announcing it
  // to the queue again will immediately cancel the synth and new content will be
  // spoken. Otherwise, new content for this utterance will be spoken whenever the old
  // content has finished speaking
  cancelSelf: true,

  // {boolean} - Only applies to two Utterances with the same priority. If true and another Utterance is currently
  // being spoken by the speech synth (or queued by voicingManager), announcing this Utterance will immediately cancel
  // the other content being spoken by the synth. Otherwise, content for the new utterance will be spoken as soon as
  // the browser finishes speaking the utterances in front of it in line.
  cancelOther: true,

  // {number} - Used to determine which utterance might interrupt another utterance. Any utterance (1) with a higher
  // priority than another utterance (2) will behave as such:
  // - (1) will interrupt (2) when (2) is currently being spoken, and (1) is announced by the voicingManager. In this
  //       case, (2) is interrupted, and never finished.
  // - (1) will continue speaking if (1) was speaking, and (2) is announced by the voicingManager. In this case (2)
  //       will be spoken when (1) is done.
  priority: DEFAULT_PRIORITY
};

class VoicingManager extends Announcer {
  constructor() {
    super( {

      // {boolean} - All VoicingManager instances should respect responseCollector's current state.
      respectResponseCollectorProperties: true
    } );

    // @public {null|SpeechSynthesisVoice}
    this.voiceProperty = new Property( null );

    // @public {NumberProperty} - controls the speaking rate of Web Speech
    this.voiceRateProperty = new NumberProperty( 1.0, { range: new Range( 0.75, 2 ) } );

    // {NumberProperty} - controls the pitch of the synth
    this.voicePitchProperty = new NumberProperty( 1.0, { range: new Range( 0.5, 2 ) } );

    // @private {boolean} - Indicates whether or not speech using SpeechSynthesis has been requested at least once.
    // The first time speech is requested, it must be done synchronously from user input with absolutely no delay.
    // requestSpeech() generally uses a timeout to workaround browser bugs, but those cannot be used until after the
    // first request for speech.
    this.hasSpoken = false;

    // @private {number} - In ms, how long to go before "waking the SpeechSynthesis" engine to keep speech
    // fast on Chromebooks, see documentation around ENGINE_WAKE_INTERVAL.
    this.timeSinceWakingEngine = 0;

    // @private {number} - Amount of time in ms to wait between speaking SpeechSynthesisUtterances, see
    // VOICING_UTTERANCE_INTERVAL for details about why this is necessary.
    this.timeSinceUtteranceEnd = 0;

    // @public {Emitter} - emits events when the speaker starts/stops speaking, with the Utterance that is
    // either starting or stopping
    this.startSpeakingEmitter = new Emitter( { parameters: [ { valueType: 'string' }, { valueType: Utterance } ] } );
    this.endSpeakingEmitter = new Emitter( { parameters: [ { valueType: 'string' }, { valueType: Utterance } ] } );

    // @public {Emitter} - emits whenever the voices change for SpeechSynthesis
    this.voicesChangedEmitter = new Emitter();

    // @private - To get around multiple inheritance issues, create enabledProperty via composition instead, then create
    // a reference on this component for the enabledProperty
    this.enabledComponentImplementation = new EnabledComponent( {

      // initial value for the enabledProperty, false because speech should not happen until requested by user
      enabled: false,

      // phet-io
      phetioEnabledPropertyInstrumented: false
    } );

    // @public
    this.enabledProperty = this.enabledComponentImplementation.enabledProperty;

    // @public {BooleanProperty} - Controls whether Voicing is enabled in a "main window" area of the application.
    // This supports the ability to disable Voicing for the important screen content of your application while keeping
    // Voicing for surrounding UI components enabled (for example).
    this.mainWindowVoicingEnabledProperty = new BooleanProperty( true );

    // @public {DerivedProperty.<Boolean>} - Property that indicates that the Voicing feature is enabled for all areas
    // of the application.
    this.voicingFullyEnabledProperty = DerivedProperty.and( [ this.enabledProperty, this.mainWindowVoicingEnabledProperty ] );

    // @public {BooleanProperty} - Indicates whether speech is fully enabled AND speech is allowed, as specified
    // by the Property provided in initialize(). See speechAllowedProperty of initialize(). In order for this Property
    // to be true, speechAllowedProperty, enabledProperty, and mainWindowVoicingEnabledProperty must all be true.
    // Initialized in the constructor because we don't have access to all the dependency Properties until initialize.
    this.speechAllowedAndFullyEnabledProperty = new BooleanProperty( false );

    // @private {SpeechSynthesis|null} - synth from Web Speech API that drives speech, defined on initialize
    this._synth = null;

    // @public {SpeechSynthesisVoice[]} - possible voices for Web Speech synthesis
    this.voices = [];

    // @public {boolean} - is the VoicingManager initialized for use? This is prototypal so it isn't always initialized
    this.initialized = false;

    // @private {Property|DerivedProperty|null} - Controls whether or not speech is allowed with synthesis.
    // Null until initialized, and can be set by options to initialize().
    this._canSpeakProperty = null;

    // @private {function} - bound so we can link and unlink to this.canSpeakProperty when the voicingManager becomes
    // initialized.
    this.boundHandleCanSpeakChange = this.handleCanSpeakChange.bind( this );

    // @private {Utterance|null} - A reference to the utterance currently in the synth being spoken by the browser, so
    // we can determine cancelling behavior when it is time to speak the next utterance. See voicing's supported
    // announcerOptions for details.
    this.currentlySpeakingUtterance = null;

    // fixes a bug on Safari where the `start` and `end` Utterances don't fire! The
    // issue is (apparently) that Safari internally clears the reference to the
    // Utterance on speak which prevents it from firing these events at the right
    // time - fix borrowed from
    // https://stackoverflow.com/questions/23483990/speechsynthesis-api-onend-callback-not-working
    // Unfortunately, this also introduces a memory leak, we should be smarter about
    // clearing this, though it is a bit tricky since we don't have a way to know
    // when we are done with an utterance - see #215
    // Blown away regularly, don't keep a reference to it.
    this.safariWorkaroundUtterancePairs = [];
  }

  /**
   * Indicate that the voicingManager is ready for use, and attempt to populate voices (if they are ready yet). Adds
   * listeners that control speech.
   * @public
   */
  initialize( options ) {
    assert && assert( this.initialized === false, 'can only be initialized once' );
    assert && assert( this.isSpeechSynthesisSupported(), 'trying to initialize speech, but speech is not supported on this platform.' );

    options = merge( {

      // {BooleanProperty|DerivedProperty.<boolean>} - Controls whether speech is allowed with speech synthesis.
      // Combined into another DerivedProperty with this.enabledProperty so you don't have to use that as one
      // of the Properties that derive speechAllowedProperty, if you are passing in a DerivedProperty.
      speechAllowedProperty: new BooleanProperty( true )
    }, options );

    this._synth = window.speechSynthesis;

    // whether the optional Property indicating speech is allowed and the voicingManager is enabled
    this._canSpeakProperty = DerivedProperty.and( [ options.speechAllowedProperty, this.enabledProperty ] );
    this._canSpeakProperty.link( this.boundHandleCanSpeakChange );

    // Set the speechAllowedAndFullyEnabledProperty when dependency Properties update
    Property.multilink(
      [ options.speechAllowedProperty, this.voicingFullyEnabledProperty ],
      ( speechAllowed, voicingFullyEnabled ) => {
        this.speechAllowedAndFullyEnabledProperty.value = speechAllowed && voicingFullyEnabled;
      } );

    // browsers tend to generate the list of voices lazily, so the list of voices may be empty until speech is
    // first requested
    this.getSynth().onvoiceschanged = () => {
      this.populateVoices();
    };

    // try to populate voices immediately in case the browser populates them eagerly and we never get an
    // onvoiceschanged event
    this.populateVoices();

    // The control key will stop the synth from speaking if there is an active utterance. This key was decided because
    // most major screen readers will stop speech when this key is pressed
    globalKeyStateTracker.keyupEmitter.addListener( domEvent => {
      if ( KeyboardUtils.isControlKey( domEvent ) ) {
        this.cancel();
      }
    } );

    // To get Voicing to happen quickly on Chromebooks we set the counter to a value that will trigger the "engine
    // wake" interval on the next animation frame the first time we get a user gesture. See ENGINE_WAKE_INTERVAL
    // for more information about this workaround.
    const startEngineListener = () => {
      this.timeSinceWakingEngine = ENGINE_WAKE_INTERVAL;

      // Display is on the namespace but cannot be imported due to circular dependencies
      scenery.Display.userGestureEmitter.removeListener( startEngineListener );
    };
    scenery.Display.userGestureEmitter.addListener( startEngineListener );

    this.initialized = true;
  }

  /**
   * Remove an element from the utterance queue.
   * @private
   *
   * @param {UtteranceWrapper} utteranceWrapper
   * @param {UtteranceWrapper[]} queue - modified by this function!
   */
  removeFromQueue( utteranceWrapper, queue ) {

    // remove from voicingManager list after speaking
    const index = queue.indexOf( utteranceWrapper );
    assert && assert( index >= 0, 'trying to remove a utteranceWrapper that doesn\'t exist' );
    queue.splice( index, 1 );
  }

  /**
   * @override
   * @private
   * @param {number} dt - in milliseconds (not seconds)!
   * @param {UtteranceWrapper[]} queue
   */
  step( dt, queue ) {

    if ( this.initialized ) {

      // Increment the amount of time since the synth has stopped speaking the previous utterance, but don't
      // start counting up until the synth has finished speaking its current utterance.
      this.timeSinceUtteranceEnd = this.getSynth().speaking ? 0 : this.timeSinceUtteranceEnd + dt;

      // Wait until VOICING_UTTERANCE_INTERVAL to speak again for more consistent behavior on certain platforms,
      // see documentation for the constant for more information. By setting readyToSpeak in the step function
      // we also don't have to rely at all on the SpeechSynthesisUtterance 'end' event, which is inconsistent on
      // certain platforms.
      if ( this.timeSinceUtteranceEnd > VOICING_UTTERANCE_INTERVAL ) {
        this.readyToSpeak = true;
      }

      // If our queue is empty and the synth isn't speaking, then clear safariWorkaroundUtterancePairs to prevent memory leak.
      // This handles any uncertain cases where the "end" callback on SpeechSynthUtterance isn't called.
      if ( !this.getSynth().speaking && queue.length === 0 && this.safariWorkaroundUtterancePairs.length > 0 ) {
        this.safariWorkaroundUtterancePairs = [];
      }

      // A workaround to keep SpeechSynthesis responsive on Chromebooks. If there is a long enough interval between
      // speech requests, the next time SpeechSynthesis is used it is very slow on Chromebook. We think the browser
      // turns "off" the synthesis engine for performance. If it has been long enough since using speech synthesis and
      // there is nothing to speak in the queue, requesting speech with empty content keeps the engine active.
      // See https://github.com/phetsims/gravity-force-lab-basics/issues/303.
      this.timeSinceWakingEngine += dt;
      if ( !this.getSynth().speaking && queue.length === 0 && this.timeSinceWakingEngine > ENGINE_WAKE_INTERVAL ) {
        this.timeSinceWakingEngine = 0;
        this.getSynth().speak( new SpeechSynthesisUtterance( '' ) );
      }
    }
  }

  /**
   * When we can no longer speak, cancel all speech to silence everything.
   * @private
   *
   * @param {boolean} canSpeak
   */
  handleCanSpeakChange( canSpeak ) {
    if ( !canSpeak ) { this.cancel(); }
  }

  /**
   * Update the list of voices available to the synth, and notify that the list has changed.
   * @private
   */
  populateVoices() {

    // the browser sometimes provides duplicate voices, prune those out of the list
    this.voices = _.uniqBy( this.getSynth().getVoices(), voice => voice.name );
    this.voicesChangedEmitter.emit();
  }

  /**
   * Returns an array of SpeechSynthesisVoices that are sorted such that the best sounding voices come first.
   * As of 9/27/21, we find that the "Google" voices sound best while Apple's "Fred" sounds the worst so the list
   * will be ordered to reflect that. This way "Google" voices will be selected by default when available and "Fred"
   * will almost never be the default Voice since it is last in the list. See
   * https://github.com/phetsims/scenery/issues/1282/ for discussion and this decision.
   * @public
   *
   * @returns {SpeechSynthesisVoice[]}
   */
  getPrioritizedVoices() {
    assert && assert( this.initialized, 'No voices available until the voicingManager is initialized' );
    assert && assert( this.voices.length > 0, 'No voices available to provided a prioritized list.' );

    return this.voices.slice().sort( ( a, b ) => {
      return a.name.includes( 'Fred' ) ? 1 : // a includes 'Fred', put b before a so 'Fred' is at the bottom
             b.name.includes( 'Fred' ) ? -1 : // b includes 'Fred', put a before b so 'Fred' is at the bottom
             a.name.includes( 'Google' ) ? -1 : // a includes 'Google', put a before b so 'Google' is at the top
             b.name.includes( 'Google' ) ? 1 : // b includes 'Google, 'put b before a so 'Google' is at the top
             0; // otherwise all voices are considered equal
    } );
  }

  /**
   * Implements announce so the voicingManager can be a source of output for utteranceQueue.
   * @public
   * @override
   *
   * @param {Utterance} utterance
   * @param {Object} [options]
   */
  announce( utterance, options ) {
    if ( this.initialized ) {
      this.speak( utterance );
    }
  }

  /**
   * Use speech synthesis to speak an utterance. No-op unless voicingManager is initialized and enabled and
   * other output controlling Properties are true (see speechAllowedProperty in initialize()).
   * @public
   *
   * @param {Utterance} utterance
   */
  speak( utterance ) {
    if ( this.initialized && this._canSpeakProperty.value ) {
      this.requestSpeech( utterance );
    }
  }

  /**
   * Use speech synthesis to speak an utterance. No-op unless voicingManager is initialized and other output
   * controlling Properties are true (see speechAllowedProperty in initialize()). This explicitly ignores
   * this.enabledProperty, allowing speech even when voicingManager is disabled. This is useful in rare cases, for
   * example when the voicingManager recently becomes disabled by the user and we need to announce confirmation of
   * that decision ("Voicing off" or "All audio off").
   * @public
   *
   * @param {Utterance} utterance
   */
  speakIgnoringEnabled( utterance ) {
    if ( this.initialized ) {
      this.requestSpeech( utterance );
    }
  }

  /**
   * Request speech with SpeechSynthesis.
   * @private
   *
   * @param {Utterance} utterance
   */
  requestSpeech( utterance ) {
    assert && assert( this.isSpeechSynthesisSupported(), 'trying to speak with speechSynthesis, but it is not supported on this platform' );

    // embedding marks (for i18n) impact the output, strip before speaking
    const stringToSpeak = removeBrTags( stripEmbeddingMarks( utterance.getTextToAlert( this.respectResponseCollectorProperties ) ) );
    const speechSynthUtterance = new SpeechSynthesisUtterance( stringToSpeak );
    speechSynthUtterance.voice = this.voiceProperty.value;
    speechSynthUtterance.pitch = this.voicePitchProperty.value;
    speechSynthUtterance.rate = this.voiceRateProperty.value;

    // keep a reference to WebSpeechUtterances in Safari, so the browser doesn't dispose of it before firing, see #215
    const utterancePair = new UtterancePair( utterance, speechSynthUtterance );
    this.safariWorkaroundUtterancePairs.push( utterancePair );

    const startListener = () => {
      this.startSpeakingEmitter.emit( stringToSpeak, utterance );
      this.currentlySpeakingUtterance = utterance;
      speechSynthUtterance.removeEventListener( 'start', startListener );
    };

    const endListener = () => {
      this.endSpeakingEmitter.emit( stringToSpeak, utterance );
      speechSynthUtterance.removeEventListener( 'end', endListener );

      // remove the reference to the SpeechSynthesisUtterance so we don't leak memory
      const indexOfPair = this.safariWorkaroundUtterancePairs.indexOf( utterancePair );
      if ( indexOfPair > -1 ) {
        this.safariWorkaroundUtterancePairs.splice( indexOfPair, 1 );
      }

      this.currentlySpeakingUtterance = null;
    };

    speechSynthUtterance.addEventListener( 'start', startListener );
    speechSynthUtterance.addEventListener( 'end', endListener );

    // In Safari the `end` listener does not fire consistently, (especially after cancel)
    // but the error event does. In this case signify that speaking has ended.
    speechSynthUtterance.addEventListener( 'error', endListener );

    // Signify to the utterance-queue that we cannot speak yet until this utterance has finished
    this.readyToSpeak = false;

    this.getSynth().speak( speechSynthUtterance );

    if ( !this.hasSpoken ) {
      this.hasSpoken = true;
    }
  }

  /**
   * Returns true if SpeechSynthesis is available on the window. This check is sufficient for all of
   * voicingManager. On platforms where speechSynthesis is available, all features of it are available, with the
   * exception of the onvoiceschanged event in a couple of platforms. However, the listener can still be set
   * without issue on those platforms so we don't need to check for its existence. On those platforms, voices
   * are provided right on load.
   * @public
   *
   * @returns {boolean}
   */
  isSpeechSynthesisSupported() {
    return !!window.speechSynthesis && !!window.SpeechSynthesisUtterance;
  }

  /**
   * Returns a references to the SpeechSynthesis of the voicingManager that is used to request speech with the Web
   * Speech API. Every references has a check to ensure that the synth is available.
   * @private
   *
   * @returns {null|SpeechSynthesis}
   */
  getSynth() {
    assert && assert( this.isSpeechSynthesisSupported(), 'Trying to use SpeechSynthesis, but it is not supported on this platform.' );
    return this._synth;
  }

  /**
   * Stops any current speech and removes all utterances that may be queued.
   * @public
   */
  cancel() {
    if ( this.initialized ) {

      // Cancel anything that is being spoken currently.
      this.cancelSynth();

      // indicate to utteranceQueues that we expect everything queued for voicing to be removed
      this.clearEmitter.emit();

      // cancel clears all utterances from the utteranceQueue, so we should clear all of the safari workaround
      // references as well
      this.safariWorkaroundUtterancePairs = [];
    }
  }

  /**
   * Given one utterance, should it cancel another provided utterance?
   * @param {Utterance} utterance
   * @param {Utterance} utteranceToCancel
   * @returns {boolean}
   * @private
   */
  shouldCancel( utterance, utteranceToCancel ) {
    assert && assert( utterance instanceof Utterance );
    assert && assert( utteranceToCancel instanceof Utterance );

    const utteranceOptions = merge( {}, UTTERANCE_OPTION_DEFAULTS, utterance.announcerOptions );
    const utteranceToCancelOptions = merge( {}, UTTERANCE_OPTION_DEFAULTS, utteranceToCancel.announcerOptions );

    let shouldCancel;
    if ( utteranceToCancelOptions.priority !== utteranceOptions.priority ) {
      shouldCancel = utteranceToCancelOptions.priority < utteranceOptions.priority;
    }
    else {
      shouldCancel = utteranceOptions.cancelOther;
      if ( utteranceToCancel && utteranceToCancel === utterance ) {
        shouldCancel = utteranceOptions.cancelSelf;
      }
    }

    return shouldCancel;
  }

  /**
   * Remove earlier Utterances from the queue if the Utterance is important enough. This will also interrupt
   * the utterance that is currently being spoken.
   * @public
   * @override
   *
   * @param newUtterance {Utterance}
   * @param {UtteranceWrapper[]} queue - The queue of the utteranceQueue. Will be modified as we prioritize!
   */
  prioritizeUtterances( newUtterance, queue ) {

    // Update the queue before canceling the browser queue, since that will most likely trigger the end
    // callback (and therefore the next utterance to be spoken).
    for ( let i = queue.length - 1; i >= 0; i-- ) {

      // {UtteranceWrapper} of UtteranceQueue
      const utteranceWrapper = queue[ i ];

      if ( this.shouldCancel( newUtterance, utteranceWrapper.utterance ) ) {
        this.removeFromQueue( utteranceWrapper, queue );

        // remove from safari workaround list to avoid memory leaks, if available
        const index = _.findIndex( this.safariWorkaroundUtterancePairs, utterancePair => utterancePair.utterance === utteranceWrapper.utterance );
        if ( index > -1 ) {
          this.safariWorkaroundUtterancePairs.splice( index, 1 );
        }
      }
    }

    // test against what is currently being spoken by the synth (currentlySpeakingUtterance)
    if ( this.currentlySpeakingUtterance && this.shouldCancel( newUtterance, this.currentlySpeakingUtterance ) ) {
      this.cancelSynth();
    }
  }

  /**
   * Cancel the synth. This will silence speech. This will silence any speech and cancel the
   * @private
   */
  cancelSynth() {
    assert && assert( this.initialized, 'must be initialized to use synth' );
    this.getSynth().cancel();
  }
}

/**
 * An inner class that pairs a SpeechSynthesisUtterance with an Utterance. Useful for the Safari workaround
 */
class UtterancePair {

  /**
   * @param {Utterance} utterance
   * @param {SpeechSynthesisUtterance} speechSynthesisUtterance
   */
  constructor( utterance, speechSynthesisUtterance ) {

    // @public (read-only)
    this.utterance = utterance;
    this.speechSynthesisUtterance = speechSynthesisUtterance;
  }
}

/**
 * @param {Object} element - returned from himalaya parser, see documentation for details.
 * @returns {boolean}
 */
const isNotBrTag = element => !( element.type.toLowerCase() === 'element' && element.tagName.toLowerCase() === 'br' );

/**
 * Remove <br> or <br/> tags from a string
 * @param {string} string - plain text or html string
 * @returns {string}
 */
function removeBrTags( string ) {
  const parsedAndFiltered = himalaya.parse( string ).filter( isNotBrTag );
  return himalaya.stringify( parsedAndFiltered );
}

const voicingManager = new VoicingManager();

// @public - Priority levels that can be used by Utterances providing the `announcerOptions.priority` option.
voicingManager.TOP_PRIORITY = 10;
voicingManager.HIGH_PRIORITY = 5;
voicingManager.MEDIUM_PRIORITY = 2;
voicingManager.DEFAULT_PRIORITY = DEFAULT_PRIORITY;
voicingManager.LOW_PRIORITY = 0;

scenery.register( 'voicingManager', voicingManager );
export default voicingManager;