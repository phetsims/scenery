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
import stepTimer from '../../../../axon/js/stepTimer.js';
import Range from '../../../../dot/js/Range.js';
import merge from '../../../../phet-core/js/merge.js';
import stripEmbeddingMarks from '../../../../phet-core/js/stripEmbeddingMarks.js';
import Announcer from '../../../../utterance-queue/js/Announcer.js';
import Utterance from '../../../../utterance-queue/js/Utterance.js';
import scenery from '../../scenery.js';
import globalKeyStateTracker from '../globalKeyStateTracker.js';
import KeyboardUtils from '../KeyboardUtils.js';

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

  // {number} - Used to determine which utterance might interrupt another utterance. Any utterance (1) with a higher priority
  // than another utterance (2) will behave as such:
  // - (1) will interrupt (2) when (2) is currently being spoken, and (1) is announced by the voicingManager. In this case, (2) is interrupted, and never finished.
  // - (1) will continue speaking if (1) was speaking, and (2) is announced by the voicingManager. In this case (2) will be spoken (1) is done.
  priority: 1
};


class VoicingManager extends Announcer {
  constructor() {
    super( {

      // All VoicingManager instances should respect responseCollector's current state.
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

    // @public {Emitter} - emits events when the speaker starts/stops speaking, with the Utterance that is
    // either starting or stopping
    this.startSpeakingEmitter = new Emitter( { parameters: [ { valueType: 'string' }, { valueType: Utterance } ] } );
    this.endSpeakingEmitter = new Emitter( { parameters: [ { valueType: 'string' }, { valueType: Utterance } ] } );

    // @public {Emitter} - emits whenever the voices change for SpeechSynthesis
    this.voicesChangedEmitter = new Emitter();

    // @public - whether or not the synth is speaking - perhaps this should
    // replace the emitters above?
    this.speakingProperty = new BooleanProperty( false );

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
    this.safariWorkaroundUtterances = [];

    // Blown away regularly, don't keep a reference to it.
    this.voicingQueue = [];
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

    // No dispose, as this singleton exists for the lifetime of the runtime.
    stepTimer.addListener( this.stepQueue.bind( this ) );

    this.initialized = true;
  }

  /**
   * Remove an element from the voicingQueue
   * @private
   * @param {VoicingQueueElement} voicingQueueElement
   */
  removeFromVoicingQueue( voicingQueueElement ) {

    // remove from voicingManager list after speaking
    const index = voicingManager.voicingQueue.indexOf( voicingQueueElement );
    assert && assert( index >= 0, 'trying to remove a voicingQueueElement that doesn\'t exist' );
    voicingManager.voicingQueue.splice( index, 1 );
  }


  /**
   * IF there is an element in the queue that has been in long enough to support the safari workaround, then alert the first
   * one.
   * @private
   */
  alertNow() {
    const synth = voicingManager.getSynth();
    if ( synth ) {
      for ( let i = 0; i < this.voicingQueue.length; i++ ) {
        const voicingQueueElement = this.voicingQueue[ i ];

        // if minTimeInQueue is zero, it should be alerted synchronously by calling alertNow
        if ( voicingQueueElement.timeInQueue >= voicingQueueElement.minTimeInQueue ) {

          synth.speak( voicingQueueElement.speechSynthUtterance );

          // remove from voicingManager list after speaking
          this.removeFromVoicingQueue( voicingQueueElement );
          break;
        }
      }

    }
  }

  /**
   * @private
   */
  onSpeechSynthesisUtteranceEnd() {
    this.alertNow();
  }

  /**
   * @private
   * @param {number} dt
   */
  stepQueue( dt ) {

    if ( this.initialized ) {

      // increase the time each element has spent in queue
      for ( let i = 0; i < this.voicingQueue.length; i++ ) {
        const voicingQueueElement = this.voicingQueue[ i ];
        voicingQueueElement.timeInQueue += dt * 1000;
      }

      // This manages the case where the 'end' event came from an utterance, but there was no next utterance ready to be
      // spoken. Make sure that we support anytime that utterances are ready but there is no "end" callback that would
      // trigger `alertNow()`.
      if ( !this.getSynth().speaking && this.voicingQueue.length > 0 ) {
        this.alertNow();
      }

      // If our queue is empty and the synth isn't speaking, then clear safariWorkaroundUtterances to prevent memory leak.
      // This handles any uncertain cases where the "end" callback on SpeechSynthUtterance isn't called.
      if ( !this.getSynth().speaking && this.voicingQueue.length === 0 && this.safariWorkaroundUtterances.length > 0 ) {
        this.safariWorkaroundUtterances = [];
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

    // TODO: likely this will need to go, but it is nice to think about the potential for these to be aligned. Perhaps there is another place this could go after the async part of queue stepping, https://github.com/phetsims/scenery/issues/1288
    // assert && assert( this.speakingProperty.value === this.getSynth().speaking, 'isSpeaking discrepancy' );

    // only cancel the previous alert if there is something new to speak
    if ( utterance.alert ) {
      this.cleanUpAndPotentiallyCancelOthers( utterance );
    }

    // embedding marks (for i18n) impact the output, strip before speaking
    const stringToSpeak = stripEmbeddingMarks( utterance.getTextToAlert( this.respectResponseCollectorProperties ) );
    const speechSynthUtterance = new SpeechSynthesisUtterance( stringToSpeak );
    speechSynthUtterance.voice = this.voiceProperty.value;
    speechSynthUtterance.pitch = this.voicePitchProperty.value;
    speechSynthUtterance.rate = this.voiceRateProperty.value;

    // keep a reference to WebSpeechUtterances in Safari, so the browser doesn't dispose of it before firing, see #215
    this.safariWorkaroundUtterances.push( speechSynthUtterance );

    const startListener = () => {
      this.startSpeakingEmitter.emit( stringToSpeak, utterance );
      this.currentlySpeakingUtterance = utterance;
      this.speakingProperty.set( true );
      speechSynthUtterance.removeEventListener( 'start', startListener );
    };

    const endListener = () => {

      // End is immediately called if no use input has occurred in a webpage
      if ( this.hasSpoken ) {

        this.endSpeakingEmitter.emit( stringToSpeak, utterance );
        this.speakingProperty.set( false );
        speechSynthUtterance.removeEventListener( 'end', endListener );

        // remove the reference to the SpeechSynthesisUtterance so we don't leak memory
        const indexOfUtterance = this.safariWorkaroundUtterances.indexOf( speechSynthUtterance );
        if ( indexOfUtterance > -1 ) {
          this.safariWorkaroundUtterances.splice( indexOfUtterance, 1 );
        }

        this.currentlySpeakingUtterance = null;

        // kick off the next element now that this one is done.
        this.onSpeechSynthesisUtteranceEnd();
      }
    };

    speechSynthUtterance.addEventListener( 'start', startListener );
    speechSynthUtterance.addEventListener( 'end', endListener );

    // In Safari the `end` listener does not fire consistently, (especially after cancel)
    // but the error event does. In this case signify that speaking has ended.
    speechSynthUtterance.addEventListener( 'error', endListener );

    const options = this.hasSpoken ? null : { minTimeInQueue: 0 };

    // Create and add the utterance to a queue which will request speech from SpeechSynthesis behind a small delay
    // (as a workaround for Safari), see VoicingQueueElement.minTimeInQueue for details.
    const voicingQueueElement = new VoicingQueueElement( utterance, speechSynthUtterance, options );
    this.voicingQueue.push( voicingQueueElement );

    if ( !this.hasSpoken ) {
      this.alertNow();

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
   * Stops any current speech and removes all utterances in the queue internal to the SpeechSynthesis
   * (not the UtteranceQueue).
   * @public
   */
  cancel() {
    if ( this.initialized ) {

      // Cancel anything that is being spoken currently.
      this.getSynth().cancel();

      // clear everything queued to be voiced.
      this.voicingQueue = [];

      // cancel clears all utterances from the internal SpeechSynthsis queue so we should
      // clear all of our references as well
      this.safariWorkaroundUtterances = [];
    }
  }

  /**
   * Given one utterance, should it cancel another provided utterance?
   * @param {Utterance} myUtterance
   * @param {Utterance} potentialToCancelUtterance
   * @returns {boolean}
   * @private
   */
  cancelThisOneQuestionMark( myUtterance, potentialToCancelUtterance ) {
    assert && assert( myUtterance instanceof Utterance );
    assert && assert( potentialToCancelUtterance instanceof Utterance );

    const myUtteranceOptions = merge( {}, UTTERANCE_OPTION_DEFAULTS, myUtterance.announcerOptions );
    const potentialToCancelUtteranceOptions = merge( {}, UTTERANCE_OPTION_DEFAULTS, potentialToCancelUtterance.announcerOptions );

    let shouldCancel;
    if ( potentialToCancelUtteranceOptions.priority !== myUtteranceOptions.priority ) {
      shouldCancel = potentialToCancelUtteranceOptions.priority < myUtteranceOptions.priority;
    }
    else {
      shouldCancel = myUtteranceOptions.cancelOther;
      if ( potentialToCancelUtterance && potentialToCancelUtterance === myUtterance ) {
        shouldCancel = myUtteranceOptions.cancelSelf;
      }
    }

    return shouldCancel;
  }

  // @private
  cleanUpAndPotentiallyCancelOthers( utteranceThatMayCancelOthers ) {

    if ( this.initialized ) {


      // Update our voicingQueue before canceling the browser queue, since that will most likely trigger the end
      // callback (and therefore the next utterance to be spoken).
      for ( let i = this.voicingQueue.length - 1; i >= 0; i-- ) {
        const voicingQueueElement = this.voicingQueue[ i ];

        if ( this.cancelThisOneQuestionMark( utteranceThatMayCancelOthers, voicingQueueElement.utterance ) ) {


          this.removeFromVoicingQueue( voicingQueueElement );

          // remove from safari workaround list to avoid memory leaks, if available
          const index = this.safariWorkaroundUtterances.indexOf( voicingQueueElement.speechSynthUtterance );
          this.safariWorkaroundUtterances.splice( index, 1 );
        }
      }

      if ( this.currentlySpeakingUtterance && this.cancelThisOneQuestionMark( utteranceThatMayCancelOthers, this.currentlySpeakingUtterance ) ) {

        // test against what is currently being spoken by the synth (currentlySpeakingUtterance)
        // TODO: does this call the `error` or 'end' callback. If so, won't this trigger another utterance to speak even though the current call to requestSpeech hasn't even been added to the queue yet!?!?! https://github.com/phetsims/scenery/issues/1288
        this.getSynth().cancel();
      }
    }
  }
}

/**
 * An inner class that is responsible for handling data associated with VoicingManager's internal voicingQueue.
 * Mostly this keeps timing data about how long it has been in a queue to workaround browser issues about speaking items
 * too soon.
 */
class VoicingQueueElement {

  /**
   * @param {Utterance} utterance
   * @param {SpeechSynthesisUtterance} speechSynthUtterance
   * @param {Object} [options]
   */
  constructor( utterance, speechSynthUtterance, options ) {

    options = merge( {

      // In Safari, the `start` and `end` listener does not fire consistently, especially after interruption with
      // cancel. But speaking behind a timeout/delay improves the behavior significantly. Timeout of 250 ms was
      // determined with testing to be a good value to use. Values less than 250 broke the workaround, while larger
      // values feel too sluggish. See https://github.com/phetsims/john-travoltage/issues/435
      minTimeInQueue: 250
    }, options );

    this.utterance = utterance;
    this.speechSynthUtterance = speechSynthUtterance;
    this.timeInQueue = 0;
    this.minTimeInQueue = options.minTimeInQueue;
  }
}


const voicingManager = new VoicingManager();

scenery.register( 'voicingManager', voicingManager );
export default voicingManager;